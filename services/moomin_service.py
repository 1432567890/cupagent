"""Moomin Market API client (https://api.moomin.cfd/market/v1).

Cross-market gift collection aggregator. Returns current per-market
prices (``snapshot``) and historical OHLC bars (``candles``) for a
collection. Quote asset is TON (= GRAM 1:1).

Authentication uses the ``X-API-Key`` header. Responses are cached in
Redis with short TTLs — snapshots are volatile, candle bars are
historical (append-only).

Endpoints used:

    GET /collections?limit=...              — known collections
    GET /collections/{slug}/snapshot         — newest quote per market
    GET /collections/{slug}/candles          — OHLC bars for one market

Slug resolution: lowercase alphanumeric version of the collection
prefix. ``Artisan Brick`` -> ``artisanbrick``, ``Plush Pepe`` ->
``plushpepe``.
"""

from __future__ import annotations

import json
import logging
import re
from typing import TYPE_CHECKING, Any

import aiohttp

from core.constants import (
    MOOMIN_BASE_URL,
    MOOMIN_CANDLES_TTL,
    MOOMIN_COLLECTIONS_TTL,
    MOOMIN_HTTP_TIMEOUT,
    MOOMIN_SNAPSHOT_TTL,
    REDIS_MOOMIN_KEY_PREFIX,
)
from core.exceptions import MoominError

if TYPE_CHECKING:
    from redis.asyncio import Redis

logger = logging.getLogger(__name__)

# Markets supported by the Moomin API (the ``market`` query param).
# These are the values the upstream service uses in its responses and
# accepts in the candles/observations queries.
SUPPORTED_MARKETS: frozenset[str] = frozenset(
    {"grapes", "mrkt", "portals", "getgems", "tonnel", "xgift"}
)

# Allowed candle intervals and the max lookback (in days) for each, per
# the API reference: 5m up to 31 days, 1h up to 366 days, 1d up to 1095.
_INTERVAL_MAX_DAYS: dict[str, int] = {"5m": 31, "1h": 366, "1d": 1095}
DEFAULT_INTERVAL = "1d"
DEFAULT_DAYS = 7


class MoominService:
    """Cached async client for the Moomin Market API.

    Responses are stored in Redis. Heavy fields (nanoTON strings, version
    metadata) are stripped in the projection step so the LLM receives
    only what it needs (display prices in TON, observed timestamps).
    """

    def __init__(
        self,
        *,
        api_key: str,
        redis: Redis,
        base_url: str = MOOMIN_BASE_URL,
    ) -> None:
        """Initialize the Moomin client.

        Args:
            api_key: API key sent as ``X-API-Key``.
            redis: Redis client for caching.
            base_url: API base URL (defaults to the production endpoint).
        """
        self._api_key = api_key
        self._redis = redis
        self._base_url = base_url.rstrip("/")
        self._session: aiohttp.ClientSession | None = None

    @property
    def session(self) -> aiohttp.ClientSession:
        """Lazily-initialized HTTP session with the X-API-Key header."""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=MOOMIN_HTTP_TIMEOUT),
                headers={
                    "X-API-Key": self._api_key,
                    "Accept": "application/json",
                },
            )
        return self._session

    # ── Slug helpers ──────────────────────────────────────────────────

    @staticmethod
    def slugify(name: str) -> str:
        """Normalize a collection name to its Moomin slug.

        ``"Artisan Brick"`` -> ``"artisanbrick"``, ``"Plush Pepe"`` ->
        ``"plushpepe"``. Anything that is not ``[a-z0-9]`` is dropped,
        and the result is lowercased. Leading/trailing whitespace is
        stripped first.

        If the caller already passes a slug, it is returned unchanged.
        """
        if not name:
            return ""
        cleaned = name.strip()
        # If it already looks like a slug, return as-is (lowercased) so
        # ``artisanbrick`` / ``ArtisanBrick`` both work.
        if " " not in cleaned and re.fullmatch(r"[A-Za-z0-9]+", cleaned):
            return cleaned.lower()
        return re.sub(r"[^a-z0-9]+", "", cleaned.lower())

    # ── Low-level GET ─────────────────────────────────────────────────

    async def _get(
        self, path: str, params: dict[str, Any] | None = None,
    ) -> Any:
        """Perform a GET request and return parsed JSON.

        Raises:
            MoominError: on network failure or a non-200 response.
                401/403 are auth/method errors; 404 is an unknown slug
                (surfaced as a distinct message); 429/503 and 5xx are
                transient — the caller backs off rather than retrying
                aggressively.
        """
        url = f"{self._base_url}{path}"
        try:
            async with self.session.get(url, params=params) as resp:
                if resp.status == 404:
                    raise MoominError("collection or resource not found")
                if resp.status != 200:
                    body = await resp.text()
                    logger.warning(
                        "Moomin: GET %s HTTP %d: %s",
                        path, resp.status, body[:200],
                    )
                    if resp.status in (401, 403):
                        raise MoominError(
                            f"auth error ({resp.status}): {body[:200]}"
                        )
                    if resp.status in (429, 503):
                        raise MoominError(
                            f"service busy ({resp.status}); back off"
                        )
                    raise MoominError(f"Moomin {resp.status}: {body[:200]}")
                return await resp.json(content_type=None)
        except aiohttp.ClientError as e:
            logger.warning("Moomin: GET %s network error: %s", path, e)
            raise MoominError(f"network error: {e}") from e

    # ── Cache helpers ─────────────────────────────────────────────────

    @staticmethod
    def _cache_key(kind: str, *parts: str) -> str:
        """Build a Redis key: ``cupagent:moomin:<kind>:<parts>``."""
        safe = ":".join(str(p).replace(":", "_") for p in parts if p)
        return f"{REDIS_MOOMIN_KEY_PREFIX}:{kind}:{safe}"

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

    async def get_collections(self, limit: int = 250) -> list[dict[str, Any]]:
        """Return the known collections.

        Args:
            limit: 1–250 (the API caps at 250).

        Returns:
            Compact list: ``[{slug, title, updated_at}, ...]``.
        """
        cap = max(1, min(int(limit), 250))
        key = self._cache_key("collections", str(cap))

        async def _producer() -> list[dict[str, Any]]:
            raw = await self._get("/collections", params={"limit": cap})
            items = (raw or {}).get("collections", []) if isinstance(raw, dict) else []
            return [
                {
                    "slug": c.get("slug"),
                    "title": c.get("title"),
                    "updated_at": c.get("updated_at"),
                }
                for c in items
                if isinstance(c, dict)
            ]

        result = await self._cached(key, MOOMIN_COLLECTIONS_TTL, _producer)
        return result or []

    async def get_snapshot(
        self,
        collection: str,
        *,
        include_derived: bool = False,
    ) -> dict[str, Any] | None:
        """Return the newest per-market quote for a collection.

        Args:
            collection: Display name (``"Artisan Brick"``) or slug
                (``"artisanbrick"``).
            include_derived: Include derived market sources (``best``,
                ``mrkt_fast``). Off by default — derived values are not
                direct venue quotes.

        Returns:
            Compact dict::

                {
                  "slug": "artisanbrick",
                  "title": "Artisan Brick",
                  "quote_asset": "TON",
                  "prices": [
                    {"market": "mrkt", "price": "46.37",
                     "observed_at": "..."}, ...
                  ],
                  "direct_floor": {"market": "mrkt", "price": "46.37",
                                    "observed_at": "..."},
                }

            ``None`` if the collection is unknown (404).
        """
        slug = self.slugify(collection)
        if not slug:
            return None
        key = self._cache_key(
            "snapshot", slug, "derived" if include_derived else "direct",
        )

        async def _producer() -> dict[str, Any] | None:
            raw = await self._get(
                f"/collections/{slug}/snapshot",
                params={"include_derived": "true" if include_derived else "false"},
            )
            return _project_snapshot(raw)

        return await self._cached(key, MOOMIN_SNAPSHOT_TTL, _producer)

    async def get_candles(
        self,
        collection: str,
        *,
        market: str,
        interval: str = DEFAULT_INTERVAL,
        from_dt: str | None = None,
        to_dt: str | None = None,
    ) -> dict[str, Any] | None:
        """Return OHLC bars for one market.

        Args:
            collection: Display name or slug.
            market: One of :data:`SUPPORTED_MARKETS`.
            interval: ``5m`` / ``1h`` / ``1d`` (default ``1d``).
            from_dt: ISO-8601 UTC start. When omitted, derived from
                :data:`DEFAULT_DAYS` and ``interval`` lookback.
            to_dt: ISO-8601 UTC end. When omitted, ``now``.

        Returns:
            Compact dict::

                {
                  "slug": "artisanbrick",
                  "title": "Artisan Brick",
                  "quote_asset": "TON",
                  "market": "mrkt",
                  "interval": "1d",
                  "from": "...", "to": "...",
                  "candles": [
                    {"start": "...", "open": "...", "high": "...",
                     "low": "...", "close": "...", "samples": 121}, ...
                  ],
                }

            ``None`` if the collection is unknown (404). Prices are kept
            as TON display strings (``price_ton``) — arithmetic on the
            raw nanoTON strings is the caller's responsibility.
        """
        slug = self.slugify(collection)
        if not slug:
            return None
        mkt = (market or "").strip().lower()
        if mkt not in SUPPORTED_MARKETS:
            raise MoominError(f"unsupported market: {market!r}")
        iv = interval if interval in _INTERVAL_MAX_DAYS else DEFAULT_INTERVAL

        # Build a deterministic cache key so repeated identical queries
        # hit the cache. ``from``/``to`` are part of the key on purpose.
        key = self._cache_key(
            "candles", slug, mkt, iv, from_dt or "auto", to_dt or "now",
        )

        params: dict[str, str] = {"market": mkt, "interval": iv}
        if from_dt:
            params["from"] = from_dt
        if to_dt:
            params["to"] = to_dt

        async def _producer() -> dict[str, Any] | None:
            raw = await self._get(f"/collections/{slug}/candles", params=params)
            return _project_candles(raw)

        return await self._cached(key, MOOMIN_CANDLES_TTL, _producer)

    async def close(self) -> None:
        """Close the HTTP session."""
        if self._session and not self._session.closed:
            await self._session.close()


# ── Projection helpers (strip heavy fields to minimize tokens) ────────


def _project_snapshot(raw: Any) -> dict[str, Any] | None:
    """Compact snapshot view (drops nanoTON strings, version metadata).

    Keeps ``market`` + display ``price`` (from ``price_ton``) +
    ``observed_at`` per quote, plus the ``direct_floor`` summary.
    Markets with an empty/missing price are kept but with ``price: None``
    so the caller can decide to skip them.
    """
    if not isinstance(raw, dict):
        return None
    coll = raw.get("collection") or {}
    prices = []
    for q in raw.get("prices", []) or []:
        if not isinstance(q, dict):
            continue
        prices.append({
            "market": q.get("market"),
            "price": q.get("price_ton"),
            "observed_at": q.get("observed_at"),
        })
    floor_raw = raw.get("direct_floor")
    floor = None
    if isinstance(floor_raw, dict):
        floor = {
            "market": floor_raw.get("market"),
            "price": floor_raw.get("price_ton"),
            "observed_at": floor_raw.get("observed_at"),
        }
    return {
        "slug": coll.get("slug"),
        "title": coll.get("title"),
        "quote_asset": raw.get("quote_asset", "TON"),
        "derived_markets_included": bool(raw.get("derived_markets_included")),
        "prices": prices,
        "direct_floor": floor,
    }


def _project_candles(raw: Any) -> dict[str, Any] | None:
    """Compact candles view (drops nanoTON strings, keeps display TON)."""
    if not isinstance(raw, dict):
        return None
    coll = raw.get("collection") or {}
    bars = []
    for b in raw.get("candles", []) or []:
        if not isinstance(b, dict):
            continue
        bars.append({
            "start": b.get("start"),
            "open": b.get("open_ton") or _nano_to_ton(b.get("open_nano")),
            "high": b.get("high_ton") or _nano_to_ton(b.get("high_nano")),
            "low": b.get("low_ton") or _nano_to_ton(b.get("low_nano")),
            "close": b.get("close_ton") or _nano_to_ton(b.get("close_nano")),
            "samples": b.get("samples"),
        })
    return {
        "slug": coll.get("slug"),
        "title": coll.get("title"),
        "quote_asset": raw.get("quote_asset", "TON"),
        "market": raw.get("market"),
        "interval": raw.get("interval"),
        "from": raw.get("from"),
        "to": raw.get("to"),
        "candles": bars,
    }


def _nano_to_ton(nano: Any) -> str | None:
    """Convert a nanoTON decimal string to a TON display string.

    The Moomin API returns ``price_nano`` as a decimal string in
    nanoTON (``"46370000000"``). For bar OHLC we normally have
    ``*_ton`` already, but fall back to this when only nano is present.
    """
    if nano is None:
        return None
    try:
        val = int(nano) / 1_000_000_000
    except (TypeError, ValueError):
        return None
    # Trim trailing zeros but keep at most 6 decimals.
    s = f"{val:.6f}".rstrip("0").rstrip(".")
    return s or "0"
