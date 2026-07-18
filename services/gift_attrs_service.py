"""Telegram collectible gift attributes scraper.

Resolves a specific gift instance (e.g. ``scaredcat-3387``) to its
``Model`` / ``Backdrop`` / ``Symbol`` by fetching the public preview page
at ``https://t.me/nft/<slug>-<number>`` and parsing the attributes table.

The page is a small static HTML document with a single ``<table>`` whose
rows look like::

    <tr><th>Model</th><td>Caramel <mark>1%</mark></td></tr>
    <tr><th>Backdrop</th><td>Persimmon <mark>1.5%</mark></td></tr>
    <tr><th>Symbol</th><td>Illuminati <mark>2.4%</mark></td></tr>

We extract the raw cell text and strip the trailing rarity percentage
(``Caramel 1%`` → ``Caramel``) so the cleaned names can be fed directly
into GiftWiki's ``model_name`` / ``backdrop_name`` filters.

Responses are cached in Redis (attributes are immutable) with a 24h TTL.
"""

from __future__ import annotations

import asyncio
import html as _html
import json
import logging
import re
from typing import TYPE_CHECKING, Any

import aiohttp

from core.constants import (
    GIFTATTRS_TTL,
    REDIS_GIFTATTRS_KEY_PREFIX,
    TELEGRAM_NFT_BASE_URL,
)
from core.exceptions import GiftAttrsError

if TYPE_CHECKING:
    from redis.asyncio import Redis

logger = logging.getLogger(__name__)

# A desktop User-Agent: t.me serves a richer attributes table to browsers
# than to the generic ``aiohttp`` default.
_BROWSER_UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/120.0.0.0 Safari/537.36"
)

# Row matcher: <tr><th>NAME</th><td>...value...</td></tr>
# Captures the th label and the inner HTML of the td (we strip tags later).
_ROW_RE = re.compile(
    r"<tr>\s*<th>(?P<label>[^<]+)</th>\s*<td>(?P<value>.*?)</td>\s*</tr>",
    re.IGNORECASE | re.DOTALL,
)

# Strip any inner tags (e.g. <mark>1%</mark>) and collapse whitespace.
_TAG_RE = re.compile(r"<[^>]+>")

# Strip a trailing rarity percentage: "Caramel 1%" → "Caramel".
_PERCENT_RE = re.compile(r"\s+\d+(?:\.\d+)?%\s*$")

# Attributes we care about, in lowercase (table label → output key).
_WANTED_LABELS = {"model": "model", "backdrop": "backdrop", "symbol": "symbol"}


class GiftAttrsService:
    """Async, Redis-cached scraper for a specific collectible gift.

    Given a collection slug (``scaredcat``) and a number (``3387``), fetch
    ``https://t.me/nft/scaredcat-3387`` and return the gift's cleaned
    model / backdrop / symbol. Returns ``None`` when the page does not
    describe a specific instance (404, unknown number, collection-only
    link) — a missing ``Model`` row is the signal of that.
    """

    def __init__(
        self,
        *,
        redis: Redis,
        base_url: str = TELEGRAM_NFT_BASE_URL,
    ) -> None:
        self._redis = redis
        self._base_url = base_url.rstrip("/")
        self._session: aiohttp.ClientSession | None = None

    @property
    def session(self) -> aiohttp.ClientSession:
        """Lazily-initialized HTTP session with a browser User-Agent."""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=15),
                headers={
                    "User-Agent": _BROWSER_UA,
                    "Accept": "text/html,application/xhtml+xml",
                    "Accept-Language": "en-US,en;q=0.9",
                },
            )
        return self._session

    # ── Cache helpers ─────────────────────────────────────────────────

    @staticmethod
    def _cache_key(slug: str, number: int) -> str:
        """Build a Redis key: ``oclp:giftattrs:<slug>:<number>``."""
        safe_slug = slug.replace(":", "_").lower()
        return f"{REDIS_GIFTATTRS_KEY_PREFIX}:{safe_slug}:{int(number)}"

    async def _cached(
        self, key: str, ttl: int, producer: Any,
    ) -> Any:
        """Get from cache or call ``producer()`` and store the result."""
        cached = await self._redis.get(key)
        if cached is not None:
            try:
                return json.loads(cached)
            except json.JSONDecodeError:
                pass
        result = await producer()
        # Cache only meaningful results (None == not a gift page). Missing
        # numbers are rare; a short recheck next time is cheap enough and
        # avoids pinning a 404 for 24h.
        if result is not None:
            await self._redis.set(
                key, json.dumps(result, ensure_ascii=False), ex=ttl,
            )
        return result

    # ── Public API ────────────────────────────────────────────────────

    async def get_attributes(
        self, slug: str, number: int,
    ) -> dict[str, str] | None:
        """Fetch and parse attributes for a specific gift instance.

        Args:
            slug: Collection slug as used in the t.me URL, e.g.
                ``scaredcat`` (no spaces, lowercase). The caller may pass
                a display name like ``Scared Cat`` — it is normalized
                internally.
            number: Gift number within the collection, e.g. ``3387``.

        Returns:
            Dict with ``model`` / ``backdrop`` / ``symbol`` keys (any of
            them may be absent if the page omits the row), or ``None`` if
            the page does not describe a specific gift instance.
        """
        slug = self._normalize_slug(slug)
        if not slug or number <= 0:
            return None

        key = self._cache_key(slug, number)

        async def _producer() -> dict[str, str] | None:
            html = await self._fetch_html(slug, number)
            return self._parse_attributes(html)

        return await self._cached(key, GIFTATTRS_TTL, _producer)

    # ── Internals ─────────────────────────────────────────────────────

    @staticmethod
    def _normalize_slug(slug: str) -> str:
        """Normalize a display name / slug to the t.me URL form.

        ``"Scared Cat"`` / ``"Scared_Cat"`` / ``"scared-cat"`` →
        ``"scaredcat"``. The t.me path uses the lowercased name with all
        non-alphanumerics stripped.
        """
        return re.sub(r"[^a-z0-9]+", "", slug.lower())

    async def _fetch_html(self, slug: str, number: int) -> str:
        """GET the gift preview page. Raises GiftAttrsError on network error."""
        url = f"{self._base_url}/{slug}-{int(number)}"
        try:
            async with self.session.get(url) as resp:
                # 404 or generic-redirect pages still return HTTP 200 with
                # the generic Telegram landing HTML; the absence of a
                # "Model" row (handled by the parser) is the real signal.
                return await resp.text()
        except aiohttp.ClientError as e:
            logger.warning(
                "GiftAttrs: GET %s network error: %s", url, e,
            )
            raise GiftAttrsError(f"network error: {e}") from e

    @classmethod
    def _parse_attributes(cls, html: str) -> dict[str, str] | None:
        """Extract model / backdrop / symbol from the preview HTML.

        Returns ``None`` if no ``Model`` row is present (i.e. the page is
        not a specific gift instance).
        """
        attrs: dict[str, str] = {}
        for m in _ROW_RE.finditer(html):
            label = m.group("label").strip().lower()
            key = _WANTED_LABELS.get(label)
            if key is None:
                continue
            value = _clean_cell(m.group("value"))
            if value:
                attrs[key] = value

        # No Model row → not a gift instance page (404 / collection link).
        if "model" not in attrs:
            return None
        return attrs

    async def close(self) -> None:
        """Close the HTTP session."""
        if self._session and not self._session.closed:
            await self._session.close()


def _clean_cell(raw_html: str) -> str:
    """Strip inner tags and the trailing rarity percentage."""
    text = _TAG_RE.sub(" ", raw_html)
    text = _html.unescape(text)  # &nbsp; &amp; &#39; → space/&/'/...
    text = re.sub(r"\s+", " ", text).strip()
    return _PERCENT_RE.sub("", text).strip()
