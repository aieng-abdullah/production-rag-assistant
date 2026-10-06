"""Filter publisher/page furniture that pollutes retrieval.

ResearchGate PDF exports prepend ~3 chunks of profile/nav boilerplate
("SEE PROFILE", citation counters, upload notices). They match queries on
term frequency but carry no document content — a reranked junk chunk wastes
a prompt slot (measured: ResearchGate header ranked #2 for
"what is fine tuning", pushing the intro chunk out of top-5).
"""

from loguru import logger

_BOILERPLATE_MARKERS = (
    "see discussions, stats, and author profiles",
    "see profile",
    "all content following this page was uploaded by",
    "the user has requested enhancement of the downloaded file",
    "researchgate.net/publication",
)


def is_boilerplate(text: str) -> bool:
    """True when chunk text is publisher/page furniture, not document content."""
    lowered = (text or "").lower()
    return any(marker in lowered for marker in _BOILERPLATE_MARKERS)


def drop_boilerplate(results: list[dict]) -> list[dict]:
    """Remove boilerplate chunks, preserving order.

    Degenerate corpora (every chunk filtered) return the original list —
    an empty prompt slot is worse than a furniture chunk.
    """
    kept = [r for r in results if not is_boilerplate(str(r.get("text", "")))]
    removed = len(results) - len(kept)
    if removed:
        logger.info(f"Boilerplate filter dropped {removed} of {len(results)} chunks")
    if not kept and results:
        logger.warning("All candidate chunks were boilerplate — returning unfiltered")
        return results
    return kept
