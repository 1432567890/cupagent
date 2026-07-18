"""Text normalization utilities for LLM replies.

Applies cosmetic fixes to model output before sending to the user:
  - Normalizes dashes (em-dash / en-dash → plain hyphen).
  - Removes trailing period / ellipsis from the last sentence.
  - Does NOT change case (would break names, TON, %, etc.).

Only applied to LLM reply text, not to system/technical messages.
"""

from __future__ import annotations

import re


# Normalize typographic dashes to plain hyphen-minus.
# em-dash (U+2014) and en-dash (U+2013) → ASCII hyphen (U+002D).
_DASH_RE = re.compile(r"[–—]")

# Pattern to strip trailing punctuation: one or two dots, possibly preceded
# by whitespace, but only if they appear at the very end of the visible text
# (ignoring any trailing whitespace or the ghost footer anchor tag).
_TRAILING_PERIOD_RE = re.compile(
    r"[.]\s*$"
)


def normalize_reply(text: str) -> str:
    """Normalize an LLM reply for display in Telegram.

    - Strips trailing whitespace.
    - Replaces em-dash / en-dash with a plain hyphen.
    - Removes a trailing period (``.``) or ellipsis (``...``) from the
      last visible sentence. The ghost footer anchor is excluded from
      matching.

    Args:
        text: Raw LLM reply text (may contain HTML tags).

    Returns:
        Normalized text.
    """
    text = text.rstrip()

    # Don't strip periods inside <code> blocks or URLs
    if not text:
        return text

    # Normalize typographic dashes to plain hyphen.
    text = _DASH_RE.sub("-", text)

    # Remove trailing period(s) — at most a single trailing dot
    # (preserves abbreviations like "e.g." since those have dots mid-sentence)
    # We look at the very last visible character before any HTML closing tags.
    cleaned = _TRAILING_PERIOD_RE.sub("", text)
    return cleaned.rstrip()
