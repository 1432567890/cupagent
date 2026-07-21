"""Shared LLM chat logic for both regular and guest message handlers.

Both ``chat.py`` (regular messages) and ``guest_chat.py`` (guest messages)
delegate to :func:`generate_reply` for the LLM call + history persistence,
keeping a single source of truth for prompt formatting and Redis history.

Post-processing pipeline (applied in order):
    1. Text normalization (``normalize_reply``).
    2. Markdown → HTML conversion (``_markdown_to_html``).
    3. Strip GiftWiki redirect sentences (``_strip_giftwiki_redirects``).
    4. Normalise thousand-separator spaces (``_normalize_thousand_spaces``).
    5. Wrap ``Gift Name #N`` in clickable links (``wrap_gift_links``).
    6. Wrap decimal prices in ``<code>`` (``_wrap_prices_in_code``).
    7. Ghost footer (``Powered by @GrapesMarket_Bot``).
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from collections.abc import Awaitable, Callable

from aiogram import Bot

from redis.asyncio import Redis

from bot.handlers._text_normalize import normalize_reply
from services.llm_service import LLMService

logger = logging.getLogger(__name__)

_HISTORY_KEY = "cupagent:chat_history:{user_id}"
_HISTORY_TTL = 3600  # 1 hour
_MAX_TURNS = 10  # 5 user + 5 assistant messages

_GHOST_FOOTER = (
    "\n\n<a href='https://t.me/grapesmarket_bot/market?startapp=uwu'>"
    "Powered by @GrapesMarket_Bot</a>"
)

# Combined skip regex: matches <code>...</code>, <tg-emoji>...</tg-emoji>,
# <pre>...</pre>, and any other standalone HTML tag — all as skip-regions
# for text processing (price wrapping, etc.).
# IMPORTANT: the closing '>' must be INSIDE each alternative. If it's
# outside the group, Python's regex backtracking prefers the shorter
# '[^>]+>' match and splits paired tags into open+close.
_TAG_SKIP_RE = re.compile(
    r"<(?:code>.*?</code>|pre>.*?</pre>|tg-emoji\b[^>]*>.*?</tg-emoji>|[^>]+>)",
    re.DOTALL,
)

# ── Shared UX constants for typing preview ──────────────────────────────

# Typing preview messages are localised based on the user's language_code.
# Russian is the default for ru* codes, English for everything else.


def _is_ru(lang_code: str | None) -> bool:
    """True if the language code is Russian (``ru``, ``ru-RU``)."""
    return bool(lang_code and lang_code.lower().startswith("ru"))


def thinking_text(lang_code: str | None = None) -> str:
    """Localised "thinking…" placeholder."""
    if _is_ru(lang_code):
        return "<i>думаю...</i>"
    return "<i>thinking...</i>"


def fetching_prices_text(lang_code: str | None = None) -> str:
    """Localised "fetching data…" placeholder."""
    if _is_ru(lang_code):
        return "<i>получаю данные...</i>"
    return "<i>fetching data...</i>"


def llm_error_text(lang_code: str | None = None) -> str:
    """Localised error message shown on LLM failure."""
    if _is_ru(lang_code):
        return (
            "<i>во время обработки запроса произошла непредвиденная ошибка. "
            "повторите позже через несколько минут</i>"
        )
    return (
        "<i>an unexpected error occurred while processing the request. "
        "please try again in a few minutes</i>"
    )


# Backwards-compatible module-level constants (Russian, used by any
# legacy import). New code should call the functions above with the
# user's language_code.
THINKING_TEXT = thinking_text("ru")
FETCHING_PRICES_TEXT = fetching_prices_text("ru")
LLM_ERROR_TEXT = llm_error_text("ru")


async def typing_action_loop(bot: Bot, chat_id: int) -> None:
    """Send chat_action typing every 4 seconds to keep the status alive.

    Runs until cancelled or the message is deleted.
    """
    try:
        while True:
            await bot.send_chat_action(chat_id=chat_id, action="typing")
            await asyncio.sleep(4)
    except asyncio.CancelledError:
        pass


# Pattern: decimal number that looks like a price (not inside existing tags).
# Matches N.NN or N,NN formats — typical floor prices.
#
# Date / version rejection: a price stands alone, dates and version strings
# don't. The lookbehind blocks matches that start right after a ``.`` or
# ``,`` (so the middle group of ``18.07.26`` can't match), and the
# lookahead blocks matches followed by another ``.`` + digit (so the first
# group of ``18.07.26`` can't match either). A list separator ``,`` after
# the fraction IS allowed — e.g. ``price 33.77, mrkt 41.31`` wraps both.
_PRICE_DECIMAL_RE = re.compile(
    r"(?<![<\w.,\d])"
    r"(\d{1,6}[.,]\d{1,2})"
    r"(?![\w.])"
)

# Currency words that, when following a number, mark it as a price.
# Built dynamically from the canonical alias tables in
# services/crypto_service.py — single source of truth, so adding a new
# currency alias there automatically makes it price-detectable here.
# Used to wrap bare integers (and decimals) that the decimal-only pattern
# above would otherwise miss — e.g. "2.5-3 грам", "~3 грам", "100 руб",
# "3 950 000 тг".
def _build_currency_words() -> str:
    """Build a regex alternation of known currency words/aliases.

    Pulls keys from ``crypto_service._CRYPTO_ALIASES`` and
    ``_FIAT_ALIASES`` plus ISO codes from ``_KNOWN_FIAT``. The longest
    alternatives are emitted first so the regex engine prefers full
    words over prefixes (e.g. "тонкоин" before "тон").
    """
    words: set[str] = set()
    try:
        from services import crypto_service as cs

        for table in (cs._CRYPTO_ALIASES, cs._FIAT_ALIASES):
            words.update(k.lower() for k in table.keys())
        words.update(c.lower() for c in cs._KNOWN_FIAT)
    except Exception:  # noqa: BLE001 — keep bot running if import breaks
        logger.warning(
            "failed to import crypto_service aliases for currency detection",
            exc_info=True,
        )
    # Escape regex metachars just in case and sort longest-first.
    sorted_words = sorted({re.escape(w) for w in words if w}, key=len, reverse=True)
    return "|".join(sorted_words)


_CURRENCY_WORDS = _build_currency_words()
# Number + currency word → wrap the number (and any trailing decimal part).
# Uses lookahead for the currency word so the word itself is NOT consumed
# (it stays in the output as plain text). Allows an optional separator
# (space, dash, or "~") between number and currency, so "2.5-3 грам"
# wraps BOTH numbers. Note: "2.5" is already wrapped by _PRICE_DECIMAL_RE
# above, so only the bare "3" reaches this pattern.
# The integer part allows up to 12 digits — fiat amounts in weak
# currencies (UAH, KZT, IDR, UZS) regularly reach 7-9 digits after
# conversion (e.g. "3 950 000 тг").
_PRICE_CURRENCY_RE = re.compile(
    r"(?<![<\w#])"
    r"(\d{1,12}(?:[.,]\d{1,2})?)"
    r"(?=\s*[-–~]?\s*(?:" + _CURRENCY_WORDS + r")\b)",
    re.IGNORECASE,
)

# Integer prices (no decimal part). Wrapped ONLY in price context — when the
# number is preceded by:
#   1. a marketplace name (optionally with dash/colon/space): "мркт - 1581"
#   2. a bare dash/colon + space: "- 1581"
#   3. a CamelCase collection name (2+ capitalized words): "Astral Shard 119"
# Cases 1+2 cover explicit market mentions. Case 3 covers list-style output
# where the LLM writes "Collection Price" per line without repeating the
# market name (e.g. a "top floors on GetGems" listing). It's safe because
# gift links now require '#' (so a bare integer after a name is unambiguously
# a price, not a gift number), and only multi-word CamelCase names match
# (avoiding false positives on stray capitalized words).
#
# Implementation note: we capture the prefix and emit it back unchanged in
# the substitution, keeping only the integer in the "wrapped" group. Using
# capturing groups instead of lookbehind because Python's ``re`` requires
# fixed-width lookbehinds and the alternations here are variable width.
_MARKET_NAMES = (
    r"грейпс|мркт|порталс|гетгемс|тоннел|хгифт|"
    r"grapes|mrkt|portal|getgems|tonnel|xgift"
)
# CamelCase collection name: 2-4 words, each starting uppercase. Matches
# "Astral Shard", "Diamond Ring", "Scared Cat", "B-Day Candle", etc.
_COLLECTION_NAME = r"(?:[A-Z][A-Za-z'’-]+(?:\s+[A-Z][A-Za-z'’-]+)+)"
_PRICE_INT_RE = re.compile(
    r"(?<![<\w#])"                                     # not after '<', word char, or '#'
    r"("                                               # G1 open — prefix to emit back
    r"(?:"
    r"(?:" + _MARKET_NAMES + r")(?:\s*[-–:=]\s*|\s+)"  # alt 1: market name + separator
    r"|[-–:=]\s+"                                       # alt 2: bare dash/colon + space
    r"|" + _COLLECTION_NAME + r"\s+"                   # alt 3: CamelCase collection + space
    r")"                                               # close NC group
    r")"                                               # G1 close
    r"(\d{2,7})"                                       # G2: the integer (2-7 digits)
    r"(?![\w.])"                                       # not followed by word char or dot
)

# Legacy alias kept for any external caller that imports the name. The actual
# price-wrapping logic now uses both _PRICE_DECIMAL_RE and _PRICE_INT_RE.
_PRICE_NUM_RE = _PRICE_DECIMAL_RE

# Markdown patterns that some LLMs emit even when asked for HTML.
# Convert these to HTML early in the pipeline.
# **bold** → <b>bold</b>  (must be processed before single-* patterns)
_MD_BOLD_RE = re.compile(r"\*\*(.+?)\*\*", re.DOTALL)
# __italic__ → <i>italic</i>
_MD_ITALIC_UNDERSCORE_RE = re.compile(r"__(.+?)__", re.DOTALL)
# *italic* → <i>italic</i>  (avoid matching ** which was handled above)
_MD_ITALIC_STAR_RE = re.compile(r"(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)", re.DOTALL)
# `code` → <code>code</code>
_MD_CODE_RE = re.compile(r"(?<!`)`([^`]+?)`(?!`)", re.DOTALL)

# Numbers with space as thousand separator: "150 000", "3 950 000", "395 000".
# LLMs sometimes format large numbers this way. We normalise them (remove the
# space) so downstream price-wrapping regexes match the full number and wrap it
# in a single <code> tag — instead of wrapping only the last group.
_THOUSAND_SPACE_RE = re.compile(r"(\d{1,3}) (\d{3})(?=(?: \d{3})*(?![\d]))")


def _normalize_thousand_spaces(text: str) -> str:
    """Remove spaces inside numbers used as thousand separators.

    ``150 000`` → ``150000``, ``3 950 000`` → ``3950000``.
    Applied iteratively because the regex matches one group at a time.
    Only processes text outside HTML tags.
    """
    if not text:
        return text

    def _normalize_chunk(chunk: str) -> str:
        prev: str | None = None
        while prev != chunk:
            prev = chunk
            chunk = _THOUSAND_SPACE_RE.sub(r"\1\2", chunk)
        return chunk

    result_parts: list[str] = []
    pos = 0
    for tag_m in _TAG_SKIP_RE.finditer(text):
        if tag_m.start() > pos:
            result_parts.append(_normalize_chunk(text[pos:tag_m.start()]))
        result_parts.append(tag_m.group(0))
        pos = tag_m.end()
    if pos < len(text):
        result_parts.append(_normalize_chunk(text[pos:]))
    return "".join(result_parts)


def _markdown_to_html(text: str) -> str:
    """Convert stray Markdown formatting to HTML.

    Some LLMs occasionally emit Markdown even when asked for HTML. This
    converts the common cases so the message renders correctly:
        ``**bold**``      → ``<b>bold</b>``
        ``__italic__``    → ``<i>italic</i>``
        ``*italic*``      → ``<i>italic</i>``
        `` `code` ``      → ``<code>code</code>``

    Only processes text outside existing HTML tags (so HTML already present
    is left untouched).
    """
    if not text:
        return text

    def _process_chunk(chunk: str) -> str:
        chunk = _MD_BOLD_RE.sub(r"<b>\1</b>", chunk)
        chunk = _MD_ITALIC_UNDERSCORE_RE.sub(r"<i>\1</i>", chunk)
        chunk = _MD_ITALIC_STAR_RE.sub(r"<i>\1</i>", chunk)
        chunk = _MD_CODE_RE.sub(r"<code>\1</code>", chunk)
        return chunk

    result_parts: list[str] = []
    pos = 0
    for tag_m in _TAG_SKIP_RE.finditer(text):
        if tag_m.start() > pos:
            result_parts.append(_process_chunk(text[pos:tag_m.start()]))
        result_parts.append(tag_m.group(0))
        pos = tag_m.end()
    if pos < len(text):
        result_parts.append(_process_chunk(text[pos:]))

    return "".join(result_parts)


# Regex to strip GiftWiki redirect sentences from LLM output.
# A "sentence" is text between sentence boundaries (start, ".", "!", "?",
# newline). We match any sentence containing a GiftWiki mention OR a
# redirect phrase ("кинь/проверь ... гифтвики", "там скажут/покажут").
# Sentence boundaries: start of text, or after [.!?\n] + optional whitespace.
# Body allows dots inside URLs/numbers: a sentence terminator is a "."
# followed by whitespace or end, not a "." inside "t.me/foo" or "3.14".
_SENT_BOUNDARY = r"(?:^|(?<=[.!?\n])\s+)"
_SENT_BODY = r"(?:(?![.!?\n](?:\s|\Z)).)*?"
_SENT_END = r"(?:[.!?\n](?:\s|\Z)|\Z)"

_GIFTWIKI_REDIRECT_RE = re.compile(
    _SENT_BOUNDARY
    + _SENT_BODY
    + r"(?:"
    r"@GiftWiki_Bot"
    r"|гифтвики"
    r"|GiftWiki"
    r"|там\s+(?:точно\s+|обязательно\s+)?(?:скажут|покажут|увидишь)"
    r")"
    + _SENT_BODY
    + _SENT_END,
    re.IGNORECASE,
)


def _strip_giftwiki_redirects(text: str) -> str:
    """Remove sentences that redirect the user to @GiftWiki_Bot.

    This is a hard post-processing step — even if the LLM ignores the
    system prompt prohibition, the redirect sentence is stripped from the
    final output. Operates per-sentence (split on ``.`` ``!`` ``?`` and
    newlines).
    """
    if not text:
        return text
    low = text.lower()
    if "giftwiki" not in low and "гифтвики" not in low and "там скажут" not in low and "там покажут" not in low:
        return text
    cleaned = _GIFTWIKI_REDIRECT_RE.sub("", text)
    # Collapse runs of whitespace left behind by removed sentences
    # (multiple spaces, 3+ newlines).
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    # Trim leading whitespace/newlines that may remain at the start.
    return cleaned.strip(" \t\n")


def _wrap_prices_in_code(text: str) -> str:
    """Wrap bare price numbers in ``<code>`` tags.

    Three kinds of numbers are wrapped:

    - **Decimals** (``1.97``, ``33.77``, ``42,59``) — always wrapped, since a
      decimal with 1-2 fractional digits is unambiguously a price in this bot.
    - **Numbers followed by a currency word** (``"3 грам"``, ``"100 руб"``,
      ``"2.5-3 грам"``, ``"~3 тон"``) — wrapped via ``_PRICE_CURRENCY_RE``,
      which uses a lookahead on the currency word so it catches integers and
      ranges that the decimal-only pattern misses.
    - **Integers** (``1581``, ``1580``) — wrapped ONLY when they appear in a
      marketplace price context: right after a marketplace name
      (``"мркт - 1581"``, ``"гетгемс 1580"``) or after a bare dash/colon
      (``"- 1581"``). Bare integers elsewhere (gift numbers, counts, years)
      are left alone.

    Skips numbers already inside HTML tags (including existing ``<code>``
    blocks) so nested tags are never produced.
    """
    if not text:
        return text

    def _wrap_chunk(chunk: str) -> str:
        chunk = _PRICE_DECIMAL_RE.sub(r"<code>\1</code>", chunk)
        chunk = _PRICE_CURRENCY_RE.sub(r"<code>\1</code>", chunk)
        # _PRICE_INT_RE captures the prefix (market/separator) in group 1
        # and the integer in group 2 — emit the prefix back unchanged and
        # wrap only the integer.
        chunk = _PRICE_INT_RE.sub(r"\1<code>\2</code>", chunk)
        return chunk

    # Walk the text, treating paired tags (code, tg-emoji) and single tags
    # as skip-regions. Plain text between them gets the <code> treatment.
    result_parts: list[str] = []
    pos = 0
    for skip_m in _TAG_SKIP_RE.finditer(text):
        if skip_m.start() > pos:
            result_parts.append(_wrap_chunk(text[pos:skip_m.start()]))
        result_parts.append(skip_m.group(0))
        pos = skip_m.end()
    if pos < len(text):
        result_parts.append(_wrap_chunk(text[pos:]))

    return "".join(result_parts)


def _sanitize_html_balance(text: str) -> str:
    """Emergency guard: strip <code>/<pre>/<b>/<i> if tags are unbalanced.

    Telegram's HTML parser rejects the whole message if any paired tag
    is missing its closer (``TelegramBadRequest: Can't find end tag
    corresponding to start tag "code"``). When that happens the user
    sees nothing. Rather than risk a total failure, we detect imbalance
    for the inner-text-style tags we emit and — if any is broken — drop
    that tag's open/close markers entirely so the message at least
    renders as plain text.

    Nested-entity violations (e.g. ``<b><code>x</b></code>``) are also
    caught: if a closer for tag A appears while tag B is still open,
    we strip all of A's markers.

    Does NOT touch ``<a>``, ``<tg-emoji>``, ``<blockquote>`` — those are
    either added by us in already-balanced positions or user-supplied
    and left alone.
    """
    if not text:
        return text
    # The tags we may have emitted and need to keep honest.
    tags = ("code", "pre", "b", "i", "u", "s")
    out = text
    for tag in tags:
        open_pat = re.compile(rf"<{tag}\b[^>]*>")
        close_pat = re.compile(rf"</{tag}>")
        # Walk open/close markers in order; track a stack. If at any point
        # we hit a close with empty stack (or other-tag on top), the
        # structure is broken → strip this tag entirely.
        stack: list[int] = []
        broken = False
        # Token stream of (type, start, end) for this tag's markers only.
        tokens: list[tuple[str, int, int]] = []
        for m in open_pat.finditer(out):
            tokens.append(("open", m.start(), m.end()))
        for m in close_pat.finditer(out):
            tokens.append(("close", m.start(), m.end()))
        tokens.sort(key=lambda t: t[1])
        for kind, _s, _e in tokens:
            if kind == "open":
                stack.append(1)
            else:  # close
                if not stack:
                    broken = True
                    break
                stack.pop()
        if broken or stack:
            # Imbalanced — drop all open/close markers for this tag.
            out = open_pat.sub("", out)
            out = close_pat.sub("", out)
    return out


def apply_ghost_format(text: str) -> str:
    """Full post-processing pipeline for an LLM reply.

    Order:
        1. Normalize trailing punctuation (``normalize_reply``).
        2. Convert stray Markdown to HTML (``_markdown_to_html``).
        3. Strip GiftWiki redirect sentences (``_strip_giftwiki_redirects``).
        4. Normalise thousand-separator spaces (``_normalize_thousand_spaces``).
        5. Wrap ``Gift Name #N`` in clickable links (``wrap_gift_links``).
        6. Wrap decimal prices in ``<code>`` (``_wrap_prices_in_code``).
        7. Append ghost footer.
        8. Sanitize HTML balance (emergency guard against broken tags).

    ``_strip_giftwiki_redirects`` runs BEFORE gift-link wrapping so the
    redirect sentence is removed as plain text (no risk of leaving a
    dangling ``<a>`` tag fragment).
    ``_normalize_thousand_spaces`` runs BEFORE price wrapping so numbers
    with space-separators (``150 000``, ``3 950 000``) become ``150000``,
    ``3950000`` and get wrapped in a single ``<code>`` tag.
    ``wrap_gift_links`` runs BEFORE price wrapping so the gift number
    isn't pulled into a ``<code>`` tag, and the resulting ``<a>`` link
    is treated as a single skip-region by later passes.
    The final ``_sanitize_html_balance`` step is a safety net: if any
    prior step ever produces an unbalanced ``<code>``/``<b>``/etc.,
    we strip those markers rather than letting Telegram reject the
    whole message.
    """
    normalized = normalize_reply(text)
    as_html = _markdown_to_html(normalized)
    without_redirects = _strip_giftwiki_redirects(as_html)
    without_spaces = _normalize_thousand_spaces(without_redirects)
    with_links = wrap_gift_links(without_spaces)
    with_code = _wrap_prices_in_code(with_links)
    sanitized = _sanitize_html_balance(with_code)
    return sanitized + _GHOST_FOOTER


# Match a gift name followed by a number: "Scared Cat #69", "Plush Pepe #100".
# Gift name: 1-4 CamelCase words (each starts uppercase, rest lowercase),
#            total length >= 4 chars.
# Number: MUST be preceded by '#' — numbers without '#' after a CamelCase
# name are almost always floor prices in this bot's context (e.g.
# "Astral Shard 119" = 119 GRAM, not gift #119). The '#' is mandatory.
# Group 1 = gift name, group 2 = '#', group 3 = number.
_GIFT_NUMBER_RE = re.compile(
    r"(?<![<\w/])"
    r"((?:[A-Z][a-zA-Z]{1,15}(?:\s+[A-Z][a-zA-Z]{1,15}){0,3}))"
    r"\s+(#)(\d{1,7})"
    r"(?![\w.])"  # not followed by a word char OR a dot (price like 33.77)
)


def _gift_slug(name: str) -> str:
    """Convert a display name to the t.me/nft URL slug.

    "Scared Cat" → "ScaredCat", "Plush Pepe" → "PlushPepe",
    "Lunar Snake" → "LunarSnake". Spaces are removed, case preserved.
    """
    return name.replace(" ", "")


def wrap_gift_links(text: str, *, bold: bool = True) -> str:
    """Wrap ``Gift Name #N`` mentions in clickable t.me/nft links.

    Transforms ``Scared Cat #69`` into
    ``<a href='t.me/nft/ScaredCat-69'><b>Scared Cat #69</b></a>`` when
    ``bold=True``, or ``<a href='t.me/nft/ScaredCat-69'>Scared Cat #69</a>``
    when ``bold=False`` (needed for inline/guest messages, where Telegram
    forbids nested entities).

    Skips text inside HTML tags (so existing markup is preserved) and
    skips numbers that look like prices (``N.NN``) — those are matched
    only when standing alone as an integer after the gift name.
    """
    if not text:
        return text

    inner_fmt = "<b>{}</b>" if bold else "{}"

    def _process_chunk(chunk: str) -> str:
        def _sub(m: re.Match) -> str:
            name = m.group(1)
            hash_sign = m.group(2)
            number = m.group(3)
            slug = _gift_slug(name)
            label = f"{name} {hash_sign}{number}"
            inner = inner_fmt.format(label)
            return f"<a href='t.me/nft/{slug}-{number}'>{inner}</a>"

        return _GIFT_NUMBER_RE.sub(_sub, chunk)

    result_parts: list[str] = []
    pos = 0
    for tag_m in _TAG_SKIP_RE.finditer(text):
        if tag_m.start() > pos:
            result_parts.append(_process_chunk(text[pos:tag_m.start()]))
        result_parts.append(tag_m.group(0))
        pos = tag_m.end()
    if pos < len(text):
        result_parts.append(_process_chunk(text[pos:]))
    return "".join(result_parts)


def apply_ghost_format_inline(text: str) -> str:
    """Post-processing for inline/guest messages.

    Same as :func:`apply_ghost_format` but uses non-bold gift links —
    because Telegram inline messages do not support nested HTML entities.
    """
    normalized = normalize_reply(text)
    as_html = _markdown_to_html(normalized)
    without_redirects = _strip_giftwiki_redirects(as_html)
    without_spaces = _normalize_thousand_spaces(without_redirects)
    with_links = wrap_gift_links(without_spaces, bold=False)
    with_code = _wrap_prices_in_code(with_links)
    sanitized = _sanitize_html_balance(with_code)
    return sanitized + _GHOST_FOOTER


def is_free_text(text: str | None) -> bool:
    """True for non-command text messages."""
    return bool(text) and not text.startswith("/")


# Maximum length of the reply-context text injected into the user prompt.
# ~500 tokens ≈ ~1500–2000 characters for Russian text; we use a conservative
# character limit to stay safely under the token budget.
_REPLY_CONTEXT_MAX_CHARS = 2000


def extract_reply_context(message: "Message") -> str:
    """Extract the text of the message being replied to, if any.

    When the user sends a guest message (or any message) as a reply to
    another message, that replied-to message's text provides important
    context for the LLM. This helper extracts it, strips HTML tags and
    truncates to ``_REPLY_CONTEXT_MAX_CHARS`` characters.

    Returns:
        The replied-to text (may be empty string if no reply or no text).
    """
    reply = message.reply_to_message
    if reply is None:
        return ""
    text = (reply.text or "").strip()
    if not text:
        return ""
    # Strip any HTML/Caption entities — we want plain text for context.
    text = re.sub(r"<[^>]+>", "", text).strip()
    if len(text) > _REPLY_CONTEXT_MAX_CHARS:
        text = text[:_REPLY_CONTEXT_MAX_CHARS] + "…"
    return text


def build_reply_text(user_text: str, reply_context: str) -> str:
    """Build the final user prompt including reply context.

    When the user replies to a message, we inject the replied-to text as
    an explicit ``<quoted_message>`` block BEFORE the user's own text, so
    phrases like «перескажи это сообщение» / «что тут написано?» refer
    unambiguously to the quoted block.

    Example output::

        <quoted_message>
        ну так долго не протянуть
        </quoted_message>

        перескажи текст этого сообщения

    If there is no reply context the original ``user_text`` is returned
    unchanged.
    """
    if not reply_context:
        return user_text
    return f"<quoted_message>\n{reply_context}\n</quoted_message>\n\n{user_text}"


async def _load_history(redis: Redis, user_id: int) -> list[dict[str, str]]:
    """Load recent conversation history for a user."""
    raw = await redis.get(_HISTORY_KEY.format(user_id=user_id))
    if not raw:
        return []
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return []


async def _save_history(
    redis: Redis, user_id: int, history: list[dict[str, str]]
) -> None:
    """Persist conversation history (trimmed to last N turns)."""
    trimmed = history[-_MAX_TURNS:]
    await redis.set(
        _HISTORY_KEY.format(user_id=user_id),
        json.dumps(trimmed),
        ex=_HISTORY_TTL,
    )


async def generate_reply(
    llm: LLMService,
    redis: Redis | None,
    user_id: int,
    user_text: str,
    *,
    price_service=None,
    crypto_service=None,
    giftwiki_service=None,
    gift_attrs_service=None,
    moomin_service=None,
    on_tool_call: Callable[[], Awaitable[None]] | None = None,
) -> str:
    """Run the LLM with history and persist the turn.

    Args:
        llm: Configured LLM service.
        redis: Optional Redis client for history persistence.
        user_id: Telegram user id (history is keyed per-user).
        user_text: The incoming user message.
        price_service: Optional PriceService for floor-price tool.
        crypto_service: Optional CryptoService for currency conversion tool.
        giftwiki_service: Optional GiftWikiService for monochrome tool.
        gift_attrs_service: Optional GiftAttrsService for resolving a
            specific gift number to its model/backdrop (monochrome lookup).
        moomin_service: Optional MoominService for cross-market snapshots
            and OHLC price history tools.
        on_tool_call: Optional async callback invoked when a tool is called.

    Returns:
        The raw model reply (without ghost footer).
    """
    history: list[dict[str, str]] = []
    if redis is not None:
        history = await _load_history(redis, user_id)

    reply = await llm.chat(
        user_text,
        user_id=user_id,
        history=history,
        price_service=price_service,
        crypto_service=crypto_service,
        giftwiki_service=giftwiki_service,
        gift_attrs_service=gift_attrs_service,
        moomin_service=moomin_service,
        on_tool_call=on_tool_call,
    )

    if redis is not None:
        history.append({"role": "user", "content": user_text})
        # Only persist the assistant turn if it actually produced text —
        # otherwise we'd poison future requests with empty replies.
        if reply.strip():
            history.append({"role": "assistant", "content": reply})
        await _save_history(redis, user_id, history)

    return reply
