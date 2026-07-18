"""Portal Market client — https://portal-market.com

Authentication: Authorization: tma <initData> header (initData from Kurigram).
Floor prices: GET /api/collections (floor_price field, string in GRAM).

Collection-scoped attribute floors (model + backdrop): the
``/api/collections/filters?short_names=…`` endpoint returns GLOBAL floors per
backdrop (the cheapest listing of that backdrop across ALL collections) — so
asking "Black фон на Scared Cat" yields the global Black-backdrop floor, not
the Scared-Cat-scoped one. To get collection-scoped floors we instead walk the
listings endpoint:

    GET /api/nfts/search?collection_ids=<uuid>&status=listed&offset=…&limit=50

Each listing carries ``price`` and an ``attributes`` array with ``type``/``value``
pairs for ``model``, ``symbol`` and ``backdrop``. We aggregate min(price) per
model and per backdrop across the first few pages — giving the real
collection-scoped floor (e.g. ~2000 for Scared Cat + Black, not 17.3).
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from core.constants import PORTAL_BASE_URL
from core.types import FloorPrice, MarketName
from markets.base import BaseMarketClient
from services.init_data_provider import InitDataProvider

logger = logging.getLogger(__name__)

# Cap on how many listings pages we walk per collection when computing per-model
# / per-backdrop floors (each page is 50 items). Most collections have a few
# hundred listings on Portal, so a handful of pages is enough to cover them.
# Floor aggregation only needs the cheapest few listings per attribute —
# listings are not price-sorted by the endpoint, so we must walk enough pages
# to have a representative sample.
_ATTR_MAX_PAGES = 10  # 10 × 50 = 500 listings
_ATTR_PAGE_SIZE = 50


class PortalClient(BaseMarketClient):
    """Client for the Portal marketplace API.

    Auth via ``Authorization: tma <initData>`` header.
    Stateless auth — initData is fetched from a Kurigram user session
    and sent with every request.
    Floor prices from GET /api/collections (floor_price as decimal string in GRAM).
    Collection-scoped attribute floors from GET /api/nfts/search
    (aggregated over listings of a single collection).
    """

    market_name = MarketName.PORTAL
    base_url = PORTAL_BASE_URL

    def __init__(
        self,
        init_data_provider: InitDataProvider,
        bot_username: str = "portals",
        app_short_name: str = "market",
    ) -> None:
        super().__init__(token="")
        self._provider = init_data_provider
        self._bot_username = bot_username
        self._app_short_name = app_short_name
        # Cache initData once per auth cycle (it's valid ~24h)
        self._cached_init_data: str = ""
        # Populated by fetch_floor_prices() — maps collection display name
        # (e.g. "Scared Cat") to its Portal UUID. Used by
        # fetch_collection_attributes() to resolve a name into the collection_ids
        # query param the /api/nfts/search endpoint requires.
        self._name_to_collection_id: dict[str, str] = {}

    def _auth_headers(self) -> dict[str, str]:
        return {"Authorization": f"tma {self._cached_init_data}"}

    async def authenticate(self) -> None:
        """Fetch fresh initData and verify it via GET /api/users/auth.

        Self-healing: if the cached initData is rejected (401), drop it,
        fetch a fresh one and try once more — so a stale-but-not-expired
        cached value (e.g. signature revoked server-side) doesn't lock
        the client out until the cache TTL.
        """
        self._cached_init_data = await self._provider.get_init_data(
            self._bot_username, self._app_short_name
        )
        try:
            await self._request("GET", "/api/users/auth", retry_on_401=False)
        except Exception:
            # First attempt failed — drop cache, fetch fresh initData, retry.
            await self._invalidate_auth()
            self._cached_init_data = await self._provider.get_init_data(
                self._bot_username, self._app_short_name
            )
            await self._request("GET", "/api/users/auth", retry_on_401=False)
        logger.info("Portal: authenticated via initData")

    async def _invalidate_auth(self) -> None:
        """Drop cached initData before a 401-driven re-auth.

        Called by :meth:`BaseMarketClient._request` when an API call
        returns 401 and the client is about to call :meth:`authenticate`
        again. Without this, ``get_init_data`` would hand back the very
        same (rejected) initData and the retry would loop pointlessly.
        """
        if hasattr(self._provider, "invalidate_cache"):
            await self._provider.invalidate_cache(
                self._bot_username, self._app_short_name
            )

    async def fetch_floor_prices(self) -> list[FloorPrice]:
        """Fetch floor prices for all gift collections.

        API returns ``{"collections": [...]}`` — extract the list.
        Floor price comes as a decimal string in GRAM.

        As a side effect this also refreshes :attr:`_name_to_collection_id`,
        which :meth:`fetch_collection_attributes` relies on to resolve a
        collection's display name to its Portal UUID.
        """
        data = await self._request(
            "GET",
            "/api/collections",
            params={"limit": "150", "offset": "0"},
        )
        # API wraps collections in a dict: {"collections": [...]}
        if isinstance(data, dict):
            items = data.get("collections", [])
        elif isinstance(data, list):
            items = data
        else:
            items = []

        prices: list[FloorPrice] = []
        name_to_id: dict[str, str] = {}
        for item in items:
            name = item.get("name")
            floor_str = item.get("floor_price")
            if not name:
                continue
            # Collection UUID — Portal uses it as the `collection_ids` query
            # param on /api/nfts/search. Field may be `id` or `collection_id`.
            cid = item.get("id") or item.get("collection_id") or ""
            if cid:
                name_to_id[name] = cid
            price = None
            if floor_str:
                try:
                    price = float(floor_str)
                except (ValueError, TypeError):
                    price = None
            prices.append(
                FloorPrice(
                    gift_name=name,
                    market=self.market_name,
                    price=price,
                    updated_at=datetime.now(timezone.utc),
                )
            )

        if name_to_id:
            self._name_to_collection_id = name_to_id

        logger.info("Portal: fetched %d floor prices", len(prices))
        return prices

    async def _ensure_name_to_collection_id(self) -> None:
        """Populate :attr:`_name_to_collection_id` if it's currently empty.

        Runs one pass over /api/collections (the same walk
        :meth:`fetch_floor_prices` performs, but only for the UUID map).
        """
        if self._name_to_collection_id:
            return
        data = await self._request(
            "GET",
            "/api/collections",
            params={"limit": "150", "offset": "0"},
        )
        items = (
            data.get("collections", []) if isinstance(data, dict)
            else data if isinstance(data, list)
            else []
        )
        for item in items if isinstance(items, list) else []:
            name = item.get("name")
            cid = item.get("id") or item.get("collection_id") or ""
            if name and cid:
                self._name_to_collection_id[name] = cid

    async def fetch_collection_attributes(
        self, collection_name: str,
    ) -> dict[str, list[dict[str, Any]]]:
        """Fetch floor prices for models and backdrops of a collection.

        Portal's ``/api/collections/filters`` endpoint returns GLOBAL floors
        per backdrop (cheapest listing of that backdrop across ALL collections),
        so it's useless for collection-scoped lookups. Instead we walk the
        listings endpoint for a single collection and aggregate min(price)
        per model and per backdrop ourselves:

            GET /api/nfts/search?collection_ids=<uuid>&status=listed
                                &exclude_bundled=true&offset=…&limit=50

        Each result item carries ``price`` (string in GRAM) and an
        ``attributes`` array of ``{type, value}`` pairs (model / symbol /
        backdrop). Listings are NOT price-sorted by the endpoint, so we walk
        up to ``_ATTR_MAX_PAGES`` pages and keep the cheapest price seen per
        attribute value.

        Args:
            collection_name: Display name like "Scared Cat".

        Returns:
            ``{"models": [...], "backdrops": [...]}`` where each item is
            ``{"name", "price" (GRAM float or None)}``. Sorted by price asc.
        """
        collection_id = self._name_to_collection_id.get(collection_name)
        if not collection_id:
            # Name not in the cached map — try refreshing it once before
            # giving up. Covers the case where fetch_floor_prices() hasn't
            # run yet or the collection entered the list after the last cycle.
            await self._ensure_name_to_collection_id()
            collection_id = self._name_to_collection_id.get(collection_name)
        if not collection_id:
            logger.debug(
                "Portal: no collection_id mapped for %r — skipping attributes",
                collection_name,
            )
            return {"models": [], "backdrops": []}

        model_floor: dict[str, float] = {}
        backdrop_floor: dict[str, float] = {}

        for page in range(_ATTR_MAX_PAGES):
            data = await self._request(
                "GET",
                "/api/nfts/search",
                params={
                    "offset": str(page * _ATTR_PAGE_SIZE),
                    "limit": str(_ATTR_PAGE_SIZE),
                    "collection_ids": collection_id,
                    "status": "listed",
                    "exclude_bundled": "true",
                    "premarket_status": "all",
                },
            )
            results = (
                data.get("results", []) if isinstance(data, dict) else []
            )
            if not results:
                break
            for item in results if isinstance(results, list) else []:
                price = self._parse_price(item.get("price"))
                if price is None:
                    continue
                attrs = item.get("attributes") or []
                if not isinstance(attrs, list):
                    continue
                for attr in attrs if isinstance(attrs, list) else []:
                    if not isinstance(attr, dict):
                        continue
                    attr_type = (attr.get("type") or "").lower()
                    value = attr.get("value")
                    if not value:
                        continue
                    bucket = (
                        model_floor if attr_type == "model"
                        else backdrop_floor if attr_type == "backdrop"
                        else None
                    )
                    if bucket is None:
                        continue
                    # Keep the cheapest price seen for this attribute value.
                    if value not in bucket or price < bucket[value]:
                        bucket[value] = price
            # Stop early if the endpoint reported fewer than a full page.
            if len(results) < _ATTR_PAGE_SIZE:
                break

        models = self._sorted_attribute_list(model_floor)
        backdrops = self._sorted_attribute_list(backdrop_floor)

        logger.info(
            "Portal: %s — %d models, %d backdrops (collection-scoped)",
            collection_name, len(models), len(backdrops),
        )
        return {"models": models, "backdrops": backdrops}

    @staticmethod
    def _parse_price(raw: Any) -> float | None:
        """Coerce a Portal price (string/int/float) to a GRAM float, or None."""
        if raw is None:
            return None
        try:
            return float(raw)
        except (ValueError, TypeError):
            return None

    @staticmethod
    def _sorted_attribute_list(
        floor_map: dict[str, float],
    ) -> list[dict[str, Any]]:
        """Turn a {name: price} map into a price-asc sorted list of dicts."""
        items = [{"name": n, "price": p} for n, p in floor_map.items()]
        items.sort(key=lambda m: (m["price"] is None, m["price"] or 0))
        return items
