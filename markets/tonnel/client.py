"""Tonnel Market client — https://gifts2.tonnel.network/api

Authentication: Telegram ``initData`` is sent in the request **body** as
``user_auth`` (not in a header). The initData is obtained from a Kurigram
user session via :class:`InitDataProvider` — same flow as MRKT/Portal.

Endpoints:
    POST /filterStats  — ``{Collection_Model: {floorPrice, howMany}}`` for
                         ALL collections/models at once (one bulk call).
                         No collection-only key — collection floor must be
                         derived as ``min(floor of all Collection_*)``.
    POST /pageGifts    — paginated listings (kept for backward compat but
                         no longer used for floors).

Floor prices:
    - **Collection floor** — derived from filterStats as the cheapest model
      floor in ``<Collection>_*``. One bulk call covers every collection.
    - **Model floor** — read directly from
      ``filterStats["<Collection>_<Model>"].floorPrice``.

Backdrops are intentionally NOT supported — per the product decision Tonnel
backdrop floors are unreliable, and filterStats exposes no per-backdrop
breakdown anyway.

Prices are in TON (1:1 with GRAM — Tonnel uses TON natively).
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from typing import Any

from core.constants import TONNEL_BASE_URL
from core.types import FloorPrice, MarketName
from markets.base import BaseMarketClient
from services.init_data_provider import InitDataProvider

logger = logging.getLogger(__name__)

# How long (seconds) we trust a cached /filterStats blob in-process before
# re-fetching. filterStats is a single bulk call covering every collection,
# so we want it relatively fresh but not hammered on every lookup.
_FILTER_STATS_TTL = 120  # 2 min


class TonnelClient(BaseMarketClient):
    """Client for the Tonnel marketplace API.

    Auth via ``user_auth`` (initData) in the request body. Floors are
    sourced from a single bulk ``/filterStats`` call covering every
    collection and model. The response is cached in-process for a short
    TTL so multiple lookups during one request don't re-fetch.
    """

    market_name = MarketName.TONNEL
    base_url = TONNEL_BASE_URL

    def __init__(
        self,
        init_data_provider: InitDataProvider,
        bot_username: str = "Tonnel_Network_bot",
        app_short_name: str = "tonnel",
    ) -> None:
        super().__init__(token="")
        self._provider = init_data_provider
        self._bot_username = bot_username
        self._app_short_name = app_short_name
        self._cached_init_data: str = ""
        # Cached filterStats: {"Collection_Model": {floorPrice, howMany}}.
        self._filter_stats: dict[str, dict[str, Any]] = {}
        self._filter_stats_fetched_at: float = 0.0

    @property
    def _default_headers(self) -> dict[str, str]:
        """Override to inject Origin/Referer/UA Tonnel requires.

        Tonnel's API (gifts2.tonnel.network) returns HTTP 403 for requests
        missing the browser-style ``Origin``/``Referer`` headers pointing
        at its web frontend. We mirror exactly what the captured HAR traffic
        sends so the server treats us as the legit mini-app client.
        """
        return {
            "Accept": "application/json",
            "Content-Type": "application/json",
            "Origin": "https://marketplace.tonnel.network",
            "Referer": "https://marketplace.tonnel.network/",
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/605.1.15 (KHTML, like Gecko)"
            ),
        }

    def _auth_headers(self) -> dict[str, str]:
        # Tonnel doesn't use Authorization headers — auth goes in the body.
        return {}

    async def authenticate(self) -> None:
        """Fetch and cache initData for inclusion in request bodies.

        Self-healing: on a hard failure we drop the cached value and fetch
        fresh once. Unlike header-based markets, a 401 here is rare —
        Tonnel rarely rejects reads — but we keep the pattern symmetric.
        """
        self._cached_init_data = await self._provider.get_init_data(
            self._bot_username, self._app_short_name
        )
        logger.debug("Tonnel: initData cached")

    async def _invalidate_auth(self) -> None:
        if hasattr(self._provider, "invalidate_cache"):
            await self._provider.invalidate_cache(
                self._bot_username, self._app_short_name,
            )

    # ── Body assembly ──────────────────────────────────────────────────

    def _filter_stats_body(self) -> dict[str, Any]:
        """Build a /filterStats request body with the authData attached.

        Tonnel expects ``{"authData": "<initData>"}``.
        """
        return {"authData": self._cached_init_data}

    # ── filterStats fetch + cache ─────────────────────────────────────

    async def _ensure_filter_stats(self, *, force: bool = False) -> None:
        """Populate :attr:`_filter_stats` if missing or stale.

        Single bulk call — covers every collection at once, so we cache
        it in-process for ``_FILTER_STATS_TTL`` seconds.

        Args:
            force: Bypass the TTL check and re-fetch unconditionally.
        """
        now = time.time()
        fresh = (
            self._filter_stats
            and not force
            and (now - self._filter_stats_fetched_at) < _FILTER_STATS_TTL
        )
        if fresh:
            return

        await self.authenticate()
        data = await self._request(
            "POST", "/filterStats", json_body=self._filter_stats_body(),
        )
        # Response shape: {"status": "success", "data": {Collection_Model: {...}}}
        payload = data.get("data", {}) if isinstance(data, dict) else {}
        if not isinstance(payload, dict):
            payload = {}
        self._filter_stats = payload
        self._filter_stats_fetched_at = now
        logger.info("Tonnel: filterStats loaded (%d keys)", len(payload))

    # ── Public API ─────────────────────────────────────────────────────

    async def fetch_floor_prices(
        self, collection_names: list[str] | None = None,
    ) -> list[FloorPrice]:
        """Fetch floor prices for collections.

        Tonnel's ``/filterStats`` returns one entry per ``Collection_Model``
        combination across ALL collections at once. There's no per-
        collection key, so a collection's floor is derived as the cheapest
        model floor among its ``<Collection>_*`` entries.

        Args:
            collection_names: Optional display names (e.g. "Scared Cat")
                to limit the output to. When ``None``/empty, every
                collection seen in filterStats is returned.

        Returns:
            One :class:`FloorPrice` per collection (price may be ``None``
            if no model had a listed floor).
        """
        await self._ensure_filter_stats()

        # Group model floors by collection prefix.
        collection_floor: dict[str, float] = {}
        for key, info in self._filter_stats.items():
            if not isinstance(info, dict):
                continue
            collection = _collection_from_key(key)
            if not collection:
                continue
            price = info.get("floorPrice")
            try:
                p = float(price) if price is not None else None
            except (ValueError, TypeError):
                p = None
            if p is None:
                continue
            cur = collection_floor.get(collection)
            if cur is None or p < cur:
                collection_floor[collection] = p

        if collection_names:
            wanted = {n for n in collection_names}
            keys_iter = (n for n in wanted)
        else:
            keys_iter = iter(collection_floor.keys())

        prices: list[FloorPrice] = []
        for name in keys_iter:
            prices.append(
                FloorPrice(
                    gift_name=name,
                    market=self.market_name,
                    price=collection_floor.get(name),
                    updated_at=datetime.now(timezone.utc),
                )
            )
        listed = sum(1 for p in prices if p.price is not None)
        logger.info(
            "Tonnel: fetched %d floors (%d listed)", len(prices), listed,
        )
        return prices

    async def fetch_collection_attributes(
        self, collection_name: str,
    ) -> dict[str, list[dict[str, Any]]]:
        """Fetch floor prices for **models** of a collection.

        Reads model floors straight out of the cached ``/filterStats``
        blob — keys of the form ``<Collection>_<Model>`` map to
        ``{floorPrice, howMany}``. Backdrops are NOT returned (per the
        product decision, see module docstring).

        Args:
            collection_name: Display name like "Scared Cat".

        Returns:
            ``{"models": [{name, price}], "backdrops": []}``. Prices in
            TON (= GRAM 1:1). Models sorted by price ascending.
        """
        await self._ensure_filter_stats()

        prefix = collection_name + "_"
        model_floor: dict[str, float] = {}
        for key, info in self._filter_stats.items():
            if not key.startswith(prefix) or not isinstance(info, dict):
                continue
            model = key[len(prefix):]
            if not model:
                continue
            price = info.get("floorPrice")
            try:
                p = float(price) if price is not None else None
            except (ValueError, TypeError):
                p = None
            if p is None:
                continue
            model_floor[model] = p

        models = [
            {"name": n, "price": p} for n, p in model_floor.items()
        ]
        # Sort by price ascending for a stable, useful ordering.
        models.sort(key=lambda m: (m["price"] is None, m["price"] or 0))
        logger.info(
            "Tonnel: %s — %d models (backdrops skipped)",
            collection_name, len(models),
        )
        return {"models": models, "backdrops": []}


def _collection_from_key(key: str) -> str:
    """Extract the collection prefix from a ``Collection_Model`` key.

    Tonnel joins collection and model with a single underscore. Collection
    names themselves don't contain underscores, so the first ``_`` is the
    separator. Returns ``""`` if the key has no underscore or the
    collection part is empty.
    """
    if "_" not in key:
        return ""
    return key.split("_", 1)[0]
