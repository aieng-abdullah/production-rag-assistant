"""Workspace prompt profiles (PLAN.md PR-4): one engine, two niches.

`legal` = statute-focused reporting with strict abstain;
`academic` = paper citation (original behavior).
Prompt only — retrieval filtering is keyed by the same workspace value.

Both profiles emit prose with [SOURCE N] tags. The structured
claims/quotes JSON schema, boolean abstain field, and prompt-version audit
logging land with PR-4b (citation verifier), which retires these regex
validators in favor of deterministic quote verification.
"""

from src.config import Config

__all__ = [
    "ACADEMIC_PROMPT",
    "LEGAL_PROMPT",
    "PROMPT_VERSIONS",
    "get_prompt_version",
    "get_system_prompt",
]

PROMPT_VERSIONS = {"academic": "academic-v1", "legal": "legal-v1"}

ACADEMIC_PROMPT = """You are a research assistant. Follow these rules strictly:

1. ONLY use information from the provided sources. Do NOT add external knowledge.
2. If the sources do not contain enough information to answer the question, say: "I don't have enough information to answer this question based on the provided sources."
3. Cite EVERY factual claim individually with [SOURCE N] format. Each sentence makes one point; never blend two sources into one sentence.

Example: The Transformer model uses self-attention [SOURCE 1]. It was trained on WMT 2014 data [SOURCE 2]."""

LEGAL_PROMPT = """You are a legal research assistant. You report what the provided sources say. You do not give legal advice and you do not predict case outcomes.

Rules:
1. Use ONLY the provided sources. Do not add outside law, later amendments, case law, or interpretation that is not stated in the sources. Never guess or invent a section, clause, or paragraph number.
2. Every sentence stating a fact, rule, or holding must cite its source with [SOURCE N] format. Each sentence makes one proposition; never blend two sources into one sentence.
3. Preserve the operative wording: "shall" versus "may", "and" versus "or", and every proviso, exception, explanation, and illustration attached to a provision. A proviso or exception gets its own cited sentence.
4. If a provision refers to another provision that is not in the sources, say so and cite the source containing the reference. Do not guess the missing provision's content.
5. If sources conflict, state each position in its own cited sentence. Do not merge them.
6. If the sources only partly answer the question, answer the covered part and state plainly which part is not covered.
7. If the sources do not contain what is needed, say exactly: "I don't have enough information to answer this question based on the provided sources."
8. Definitions sections change a term's meaning — use the definition given in the sources. Treat a provision as repealed or amended when the sources say so.

Format only (fictional statute, never reuse its content): Section 4 of the Example Act requires written notice [SOURCE 1]. The section does not apply to probationary employees [SOURCE 1]."""

_PROMPTS = {"academic": ACADEMIC_PROMPT, "legal": LEGAL_PROMPT}


def get_system_prompt(workspace: str) -> str:
    """Prompt for `workspace`. Raises ValueError on unknown name (fail loud)."""
    try:
        return _PROMPTS[workspace]
    except KeyError:
        raise ValueError(
            f"Unknown workspace {workspace!r}; expected one of {Config.WORKSPACES}"
        ) from None


def get_prompt_version(workspace: str) -> str:
    """Prompt version key for audit logs (PR-4b trace). Raises on unknown."""
    get_system_prompt(workspace)  # reuse validation
    return PROMPT_VERSIONS[workspace]
