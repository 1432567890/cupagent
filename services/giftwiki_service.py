"""GiftWiki API client (https://api.giftwiki.tg).

Wraps the public GiftWiki endpoints behind a Redis-cached async client.
Authentication uses ``X-API-Key`` (collection:read scope).

All responses are projected to a compact subset of fields before being
returned — this keeps LLM tool-result payloads small (no large photo /
animation / preview URLs), minimizing input tokens.

Endpoints used:
    GET /collection                — list / search public collections
    GET /collection?model_name=... — find collections by gift model_name
    GET /collection/{id}           — collection details with gifts
    GET /gifts/monochromes         — gift monochrome classification
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import TYPE_CHECKING, Any

import aiohttp

from core.constants import (
    GIFTWIKI_BASE_URL,
    GIFTWIKI_DETAIL_TTL,
    GIFTWIKI_SEARCH_TTL,
    REDIS_GIFTWIKI_KEY_PREFIX,
)
from core.exceptions import oclpError

if TYPE_CHECKING:
    from redis.asyncio import Redis

logger = logging.getLogger(__name__)


class GiftWikiError(oclpError):
    """GiftWiki API error."""


class GiftWikiService:
    """Cached async client for the GiftWiki Collections API.

    Responses are stored in Redis with separate TTLs for search (10min)
    and detail (1h). Heavy fields (URLs, timestamps) are stripped in the
    projection step so the LLM receives only what it needs.
    """

    def __init__(
        self,
        *,
        api_key: str,
        redis: Redis,
        base_url: str = GIFTWIKI_BASE_URL,
    ) -> None:
        self._api_key = api_key
        self._redis = redis
        self._base_url = base_url.rstrip("/")
        self._session: aiohttp.ClientSession | None = None

    @property
    def session(self) -> aiohttp.ClientSession:
        """Lazily-initialized HTTP session with the X-API-Key header."""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=20),
                headers={
                    "X-API-Key": self._api_key,
                    "Accept": "application/json",
                },
            )
        return self._session

    # ── Low-level GET ─────────────────────────────────────────────────

    async def _get(
        self, path: str, params: dict[str, Any] | None = None,
    ) -> Any:
        """Perform a GET request and return parsed JSON.

        Raises:
            GiftWikiError: on non-200 or network failure.
        """
        url = f"{self._base_url}{path}"
        try:
            async with self.session.get(url, params=params) as resp:
                if resp.status == 404:
                    return None
                if resp.status != 200:
                    body = await resp.text()
                    logger.warning(
                        "GiftWiki: GET %s HTTP %d: %s",
                        path, resp.status, body[:200],
                    )
                    raise GiftWikiError(
                        f"GiftWiki {resp.status}: {body[:200]}"
                    )
                return await resp.json(content_type=None)
        except aiohttp.ClientError as e:
            logger.warning("GiftWiki: GET %s network error: %s", path, e)
            raise GiftWikiError(f"network error: {e}") from e

    # ── Cache helpers ─────────────────────────────────────────────────

    @staticmethod
    def _cache_key(kind: str, *parts: str) -> str:
        """Build a Redis key: ``oclp:giftwiki:<kind>:<parts>``."""
        safe = ":".join(str(p).replace(":", "_") for p in parts if p)
        return f"{REDIS_GIFTWIKI_KEY_PREFIX}:{kind}:{safe}"

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
        if result is not None:
            await self._redis.set(
                key, json.dumps(result, ensure_ascii=False), ex=ttl,
            )
        return result

    # ── Public API ────────────────────────────────────────────────────

    async def search_collections(
        self, query: str, limit: int = 10,
    ) -> list[dict[str, Any]]:
        """Search public collections by name/keyword.

        Args:
            query: Free-text search term.
            limit: Max items to return (1-50, clamped).

        Returns:
            Compact list of collections: ``[{id, name, count,
            startapp}, ...]``. Returns ``[]`` on miss.
        """
        limit = max(1, min(50, int(limit)))
        key = self._cache_key("search", query.lower().strip(), str(limit))

        async def _producer() -> list[dict[str, Any]]:
            raw = await self._get(
                "/collection",
                params={"search": query, "limit": limit},
            )
            if not isinstance(raw, list):
                return []
            return [_project_collection_item(item) for item in raw]

        result = await self._cached(key, GIFTWIKI_SEARCH_TTL, _producer)
        return result or []

    async def lookup_by_model_name(
        self, model_name: str, limit: int = 5,
    ) -> list[dict[str, Any]]:
        """Find collections containing a gift whose slug matches ``model_name``.

        Resolves slug-style inputs (e.g. ``spicedwine`` from a user-typed
        ``spicedwine-42846``) into the canonical collection name(s) (e.g.
        ``Spiced Wine``). The GiftWiki ``/collection`` endpoint does NOT
        accept slug-style ``model_name`` (it requires the canonical name
        with spaces, e.g. ``Spiced Wine``), so this method first maps the
        slug to the canonical gift name via :meth:`_get_slug_index`, then
        queries the API with the canonical name.

        Note: not every Telegram collection has a matching GiftWiki
        collection (e.g. ``Spiced Wine`` is not a GiftWiki collection
        name), so this may return ``[]`` even for valid slugs. For
        slug → canonical-name resolution alone, use
        :meth:`resolve_canonical_name` which never misses.

        Args:
            model_name: The gift's slug as used in t.me URLs (lowercase,
                no separators: ``spicedwine``, ``scaredcat``).
            limit: Max items to return (1-50, clamped).

        Returns:
            Compact list of collections: ``[{id, name, count,
            startapp}, ...]``. Returns ``[]`` on miss.
        """
        limit = max(1, min(50, int(limit)))
        slug = model_name.strip().lower()
        if not slug:
            return []

        canonical = await self.resolve_canonical_name(slug)
        if not canonical:
            return []

        key = self._cache_key("model", canonical.lower(), str(limit))

        async def _producer() -> list[dict[str, Any]]:
            raw = await self._get(
                "/collection",
                params={"model_name": canonical, "limit": limit},
            )
            if not isinstance(raw, list):
                return []
            return [_project_collection_item(item) for item in raw]

        result = await self._cached(key, GIFTWIKI_DETAIL_TTL, _producer)
        return result or []

    async def resolve_canonical_name(self, model_name: str) -> str | None:
        """Resolve a gift slug to its canonical display name.

        ``spicedwine`` → ``Spiced Wine``, ``scaredcat`` → ``Scared Cat``.
        Uses the locally-built slug index (:meth:`_get_slug_index`).
        Returns ``None`` if the slug is unknown.
        """
        slug = model_name.strip().lower()
        if not slug:
            return None
        index = await self._get_slug_index()
        return index.get(slug)

    async def _get_slug_index(self) -> dict[str, str]:
        """Return a ``slug → canonical gift name`` map.

        Built once and cached in Redis for 24h (gift names are stable).
        Walks every public collection (paginated, 50/page) and indexes
        each gift's ``name`` by its slug (lowercase, no spaces). When
        the same slug maps to multiple names, the first one wins.

        Returns an empty dict if the build fails — callers treat a miss
        the same as "not found".
        """
        cache_key = self._cache_key("slug_index", "v1")
        cached = await self._redis.get(cache_key)
        if cached:
            try:
                return json.loads(cached)
            except json.JSONDecodeError:
                pass

        index: dict[str, str] = {}
        try:
            page = 1
            while True:
                raw = await self._get(
                    "/collection",
                    params={"limit": 50, "page": page},
                )
                if not isinstance(raw, list) or not raw:
                    break
                for collection in raw:
                    # We need each collection's gifts to get the gift
                    # names — the list endpoint does not include them,
                    # so fetch the detail. Skip on any failure.
                    cid = collection.get("_id")
                    if not cid:
                        continue
                    try:
                        detail = await self._get(f"/collection/{cid}")
                    except GiftWikiError:
                        continue
                    if not isinstance(detail, dict):
                        continue
                    for gift in detail.get("gifts") or []:
                        name = gift.get("name") or ""
                        if not name:
                            continue
                        gift_slug = name.lower().replace(" ", "")
                        # First occurrence wins; later duplicates skipped.
                        index.setdefault(gift_slug, name)
                if len(raw) < 50:
                    break
                page += 1
                # Hard safety cap — never loop forever if the API misbehaves.
                if page > 20:
                    break
        except GiftWikiError:
            logger.warning("GiftWiki: slug index build failed", exc_info=True)
            return index

        if index:
            await self._redis.set(
                cache_key, json.dumps(index, ensure_ascii=False),
                ex=86400,  # 24h
            )
        logger.info("GiftWiki: built slug index (%d gifts)", len(index))
        return index

    async def get_collection_detail(
        self, collection_id: str,
    ) -> dict[str, Any] | None:
        """Get full collection details including its gifts.

        Args:
            collection_id: 24-char hex MongoDB ObjectId.

        Returns:
            Compact dict: ``{id, name, keywords, count, gifts,
            related}``. ``None`` if not found.
        """
        key = self._cache_key("detail", collection_id)

        async def _producer() -> dict[str, Any] | None:
            raw = await self._get(f"/collection/{collection_id}")
            if raw is None:
                return None
            return _project_collection_detail(raw)

        return await self._cached(key, GIFTWIKI_DETAIL_TTL, _producer)

    async def get_monochromes(
        self,
        *,
        gift_name: str | None = None,
        model_name: str | None = None,
        backdrop_name: str | None = None,
        type_filter: str | None = None,
    ) -> list[dict[str, Any]]:
        """Fetch gift monochrome classification (low/medium/high/combo).

        Args:
            gift_name: Filter by gift name.
            model_name: Filter by model name.
            backdrop_name: Filter by backdrop name.
            type_filter: One of low/medium/high/combo.

        Returns:
            Compact list: ``[{gift_id, gift_name, model_name,
            backdrop_name, type}, ...]``.
        """
        params: dict[str, Any] = {}
        if gift_name:
            params["gift_name"] = gift_name
        if model_name:
            params["model_name"] = model_name
        if backdrop_name:
            params["backdrop_name"] = backdrop_name
        if type_filter:
            params["type"] = type_filter

        key = self._cache_key("mono", json.dumps(params, sort_keys=True))

        async def _producer() -> list[dict[str, Any]]:
            raw = await self._get("/gifts/monochromes", params=params)
            if not isinstance(raw, list):
                return []
            return [_project_monochrome(item) for item in raw]

        result = await self._cached(key, GIFTWIKI_DETAIL_TTL, _producer)
        return result or []

    async def close(self) -> None:
        """Close the HTTP session."""
        if self._session and not self._session.closed:
            await self._session.close()


# ── Projection helpers (strip heavy fields to minimize tokens) ────────

def _project_collection_item(item: dict[str, Any]) -> dict[str, Any]:
    """Compact list-item view of a collection."""
    return {
        "id": item.get("_id"),
        "name": item.get("name"),
        "count": item.get("count"),
        "startapp": item.get("startapp"),
    }


def _project_gift(gift: dict[str, Any]) -> dict[str, Any]:
    """Compact gift view inside a collection detail."""
    return {
        "name": gift.get("name"),
        "gift_id": gift.get("gift_id"),
        "model_name": gift.get("model_name"),
        "model_rarity": gift.get("model_rarity"),
        "supply_tier": gift.get("supply_tier"),
        "count": gift.get("count"),
    }


def _project_collection_detail(raw: dict[str, Any]) -> dict[str, Any]:
    """Compact detail view of a collection (drops URLs/timestamps)."""
    gifts_raw = raw.get("gifts") or []
    related_raw = raw.get("related_collections") or []
    return {
        "id": raw.get("_id"),
        "name": raw.get("name"),
        "keywords": raw.get("keywords") or [],
        "count": raw.get("count"),
        "startapp": raw.get("startapp"),
        "gifts": [_project_gift(g) for g in gifts_raw],
        "related": [
            {"name": r.get("name"), "id": r.get("_id")}
            for r in related_raw if r.get("name")
        ],
    }


def _project_monochrome(item: dict[str, Any]) -> dict[str, Any]:
    """Compact monochrome view (drops colors/photo/animation URLs)."""
    return {
        "gift_id": item.get("gift_id"),
        "gift_name": item.get("gift_name"),
        "model_name": item.get("model_name"),
        "backdrop_name": item.get("backdrop_name"),
        "type": item.get("type"),
    }
