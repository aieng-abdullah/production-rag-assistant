"""Sanitize untrusted text before it enters a delimited prompt block.

Framework-free so the citation prompt builder, the entailment judge, and
the API boundary all apply identical rules: drop C0 control characters
and NULs (keeping tab and newline), then drop case-insensitive delimiter
tags so user queries and retrieved chunks cannot break out of the block
they are wrapped in.
"""

import re

__all__ = [
    "CONTROL_CHARS_RE",
    "UNTRUSTED_TAGS",
    "sanitize_text",
    "sanitize_untrusted",
    "strip_control_chars",
    "strip_delimiters",
]

CONTROL_CHARS_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")

UNTRUSTED_TAGS = ("sources", "question", "claim", "evidence", "source", "conversation")


def strip_control_chars(text: str) -> str:
    """Remove C0 controls and NULs; tab (0x09) and newline (0x0A) survive."""
    return CONTROL_CHARS_RE.sub("", str(text))


def strip_delimiters(text: str, *tags: str) -> str:
    """Remove `<tag>`, `</tag>` and `<tag attrs>` for each tag, case-insensitive."""
    if not tags:
        return str(text)
    names = "|".join(sorted((re.escape(tag) for tag in tags), key=len, reverse=True))
    pattern = rf"</?\s*(?:{names})\b[^>]*>"
    return re.sub(pattern, "", str(text), flags=re.IGNORECASE)


def sanitize_text(text: str, *tags: str) -> str:
    """Controls first, then delimiters."""
    return strip_delimiters(strip_control_chars(text), *tags)


def sanitize_untrusted(text: str) -> str:
    """Shared entry point for every untrusted string entering a prompt block."""
    return sanitize_text(text, *UNTRUSTED_TAGS)
