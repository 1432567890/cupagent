"""GiftWiki API client (https://api.giftwiki.tg).

Wraps the GiftWiki monochrome endpoint behind a Redis-cached async client.
Authentication uses ``X-API-Key`` (gift:read scope).

Collection search / detail / slug-index endpoints were removed — the bot
now relies on its own market floor-price data for collection lookups and
keeps GiftWiki only for monochrome classification.

Endpoint used:
    GET /gifts/monochromes  — gift monochrome classification
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any

import aiohttp

from core.constants import (
    GIFTWIKI_BASE_URL,
    GIFTWIKI_DETAIL_TTL,
    REDIS_GIFTWIKI_KEY_PREFIX,
)
from core.exceptions import cupagentError

if TYPE_CHECKING:
    from redis.asyncio import Redis

logger = logging.getLogger(__name__)


class GiftWikiError(cupagentError):
    """GiftWiki API error."""


class GiftWikiService:
    """Cached async client for the GiftWiki monochrome API.

    Responses are stored in Redis. Heavy fields (URLs, timestamps) are
    stripped in the projection step so the LLM receives only what it needs.
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
        """Build a Redis key: ``cupagent:giftwiki:<kind>:<parts>``."""
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

def _project_monochrome(item: dict[str, Any]) -> dict[str, Any]:
    """Compact monochrome view (drops colors/photo/animation URLs)."""
    return {
        "gift_id": item.get("gift_id"),
        "gift_name": item.get("gift_name"),
        "model_name": item.get("model_name"),
        "backdrop_name": item.get("backdrop_name"),
        "type": item.get("type"),
    }
