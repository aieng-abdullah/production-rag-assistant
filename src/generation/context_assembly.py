"""Answer-time context assembly.

One choke point between retrieval and everything that reads the answer
context: the prompt builder, the deterministic quote verifier, the
entailment judge, and the trace.

Why this exists as its own module: those four consumers share a single
positional list, indexed by ``citation.source_id - 1``. ``schema.py``
resolves a quote against ``chunks[source_id - 1]`` and ``verifier.py``
scores a claim against the same position. If the prompt were widened but
the judge were not, the judge would verify text the user never saw —
silently, because both would still be internally consistent.

So widening happens here, once, and every consumer reads the result.

Default configuration is a strict no-op: with both widths at 0 the
assembled list is the retrieved list, same order, same length, same
objects. That is what makes this safe to land before the widening in
#125 — nothing changes until a width is actually set.
"""

from typing import List

from loguru import logger

from src.config import Config

__all__ = ["assemble_context", "is_widening_enabled"]


def is_widening_enabled() -> bool:
    """True when any widening knob is switched on."""
    return bool(Config.CONTEXT_SECTION_WIDTH or Config.CONTEXT_NEIGHBOR_WINDOW)


def _section_key(chunk: dict) -> str | None:
    """Identity of the section a chunk belongs to, if it has one.

    #124 adds section metadata; chunks stored before that carry no
    section and must degrade to the neighbour path rather than being
    dropped or grouped under a fake shared section.
    """
    doc_id = chunk.get("doc_id")
    section = chunk.get("section_id")
    if not doc_id or section is None:
        return None
    return f"{doc_id}:{section}"


def _widen_to_sections(chunks: List[dict], all_chunks: List[dict]) -> List[dict]:
    """Replace each hit with the full text of its section.

    Width is measured in sections, not characters: with width 1 the hit
    keeps its own section, with width 2 it also takes the neighbouring
    section. Sections are taken in document order so a widened chunk
    reads continuously rather than as stitched fragments.
    """
    width = Config.CONTEXT_SECTION_WIDTH
    if width <= 0:
        return list(chunks)
    by_key: dict[str, list[dict]] = {}
    for chunk in all_chunks:
        key = _section_key(chunk)
        if key is not None:
            by_key.setdefault(key, []).append(chunk)

    sections: dict[str, list[tuple[int, dict]]] = {
        key: [(index, member) for index, member in enumerate(members)]
        for key, members in by_key.items()
    }

    widened: List[dict] = []
    for chunk in chunks:
        key = _section_key(chunk)
        members = by_key.get(key) if key else None
        if not members:
            widened.append(chunk)
            continue

        positions = sections[key]
        anchor = next(
            (i for i, member in positions if member.get("chunk_id") == chunk.get("chunk_id")),
            None,
        )
        if anchor is None:
            widened.append(chunk)
            continue

        half = max(0, (width - 1) // 2)
        start = max(0, anchor - half)
        end = min(len(positions), start + width)
        start = max(0, end - width)
        selected = [member for _, member in positions[start:end]]

        merged = dict(chunk)
        merged["text"] = "\n\n".join(member.get("text", "") for member in selected)
        merged["context_widened"] = True
        merged["context_chunk_count"] = len(selected)
        widened.append(merged)
    return widened


def _widen_to_neighbors(chunks: List[dict], all_chunks: List[dict]) -> List[dict]:
    """Extend each hit with the chunks around it inside the same document.

    Used where no section metadata exists — the fallback that #124's
    sentinel case needs. Never crosses a document boundary, since
    adjacent chunks from a different statute would be misleading
    rather than merely irrelevant.
    """
    window = Config.CONTEXT_NEIGHBOR_WINDOW
    if window <= 0:
        return list(chunks)

    by_doc: dict[str, dict[str, dict]] = {}
    order: dict[str, list[str]] = {}
    for chunk in all_chunks:
        doc_id = chunk.get("doc_id")
        chunk_id = chunk.get("chunk_id")
        if not doc_id or not chunk_id:
            continue
        by_doc.setdefault(doc_id, {})[chunk_id] = chunk
        order.setdefault(doc_id, []).append(chunk_id)

    widened: List[dict] = []
    for chunk in chunks:
        doc_id = chunk.get("doc_id")
        chunk_id = chunk.get("chunk_id")
        members = by_doc.get(doc_id)
        doc_order = order.get(doc_id)
        if not members or not doc_order or chunk_id not in members:
            widened.append(chunk)
            continue

        anchor = doc_order.index(chunk_id)
        start = max(0, anchor - window)
        end = min(len(doc_order), anchor + window + 1)
        selected = [members[cid] for cid in doc_order[start:end]]

        merged = dict(chunk)
        merged["text"] = "\n\n".join(member.get("text", "") for member in selected)
        merged["context_widened"] = True
        merged["context_chunk_count"] = len(selected)
        widened.append(merged)
    return widened


def _enforce_budget(chunks: List[dict]) -> List[dict]:
    """Drop lowest-ranked chunks until the assembled text fits the budget.

    Truncation is deterministic: highest retrieval rank survives longest,
    ties broken by position. Exceeding the budget is a logged event, not
    an error — the answer is still worth returning with less context.
    """
    budget = Config.CONTEXT_CHAR_BUDGET
    if budget <= 0:
        return chunks

    kept = list(chunks)
    while kept and sum(len(chunk.get("text", "")) for chunk in kept) > budget:
        kept.pop()
    if len(kept) != len(chunks):
        logger.warning(
            f"Context budget {budget} chars exceeded — dropped "
            f"{len(chunks) - len(kept)} lowest-ranked chunk(s)"
        )
    return kept


def assemble_context(
    chunks: List[dict],
    all_chunks: List[dict] | None = None,
) -> List[dict]:
    """Build the answer-time context from retrieved chunks.

    ``chunks`` is the retrieval result, in rank order. ``all_chunks`` is
    the full corpus for the workspace, used as the source for widening;
    it is optional so callers that only have the hits can still use this
    as a no-op pass-through.

    Every consumer of the answer context — prompt, quote verification,
    entailment judge, trace — must read the output of this function, not
    the raw retrieval result.
    """
    if not chunks:
        return []

    if not is_widening_enabled() or not all_chunks:
        return chunks

    widened = _widen_to_sections(chunks, all_chunks)
    if Config.CONTEXT_NEIGHBOR_WINDOW:
        # Chunks with no section metadata fall through to the neighbour
        # path; ones already widened keep their section text.
        sectioned = {chunk.get("chunk_id") for chunk in widened if chunk.get("context_widened")}
        untouched = [c for c in chunks if c.get("chunk_id") not in sectioned]
        if untouched:
            neighbor_filled = _widen_to_neighbors(untouched, all_chunks)
            widened = _merge_widened(widened, untouched, neighbor_filled)

    return _enforce_budget(widened)


def _merge_widened(
    widened: List[dict],
    untouched: List[dict],
    neighbor_filled: List[dict],
) -> List[dict]:
    """Replace section-less hits with their neighbour-widened versions.

    Order and length must match the retrieval result exactly — citation
    ids are positional, so this cannot reorder or drop anything.
    """
    replacements = {
        original.get("chunk_id"): filled
        for original, filled in zip(untouched, neighbor_filled)
        if filled.get("context_widened")
    }
    return [replacements.get(chunk.get("chunk_id"), chunk) for chunk in widened]