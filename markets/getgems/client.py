"""GetGems Market client — https://api.getgems.io/public-api

Authentication: ``Authorization: Bearer <token>`` (a GGLLM_... token issued
by the TON Proof flow; we accept a pre-issued static token via env).

Floor prices: ``GET /v1/gifts/collections/top?kind=all`` — the top endpoint
is the only gifts-listing endpoint that exposes ``floorPrice`` directly
(the plain ``/v1/gifts/collections`` list returns collection metadata
without any price). Cursor-paginated (``after`` parameter) at 10 items/page.

GetGems names collections in the plural ("Scared Cats"); the rest of the
bot (Grapes/MRKT/Portal, the prompt, gift links) uses the singular
("Scared Cat"). Names are singularized on ingest so cross-market lookups
line up — see :func:`_singularize_collection_name`.

Per-attribute floors (model / backdrop): ``GET /v1/collection/attributes/{address}``
which returns ``minPrice`` per trait value. This endpoint takes a collection
*address* rather than a name, so :meth:`fetch_floor_prices` records a
``name → address`` map as a side effect, and :meth:`fetch_collection_attributes`
resolves names through it.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from core.constants import GETGEMS_BASE_URL
from core.types import FloorPrice, MarketName
from markets.base import BaseMarketClient

logger = logging.getLogger(__name__)

# The /gifts/collections/top endpoint returns 10 items per page and does not
# accept a limit override — we just walk the cursor until it dries up.
# Hard safety cap on how many pages we walk per fetch (avoids runaway loops
# if the cursor never terminates for some reason). 20 pages × 10 = 200.
_MAX_PAGES = 20


def _singularize_collection_name(name: str) -> str:
    """GetGems names collections in the plural ("Scared Cats"), while every
    other market (Grapes/MRKT/Portal) and the bot's prompt/links use the
    singular ("Scared Cat"). Normalize GetGems names to the singular form so
    prices and attributes line up under the same key cross-market.

    Rule: strip a trailing ``s`` unless the name ends in ``ss`` (e.g.
    "Kiss" stays) — covers 114 of 115 known collections. Irregular cases
    like "Jacks-in-the-Box" are left untouched.
    """
    if not name:
        return name
    if name.endswith("s") and not name.endswith("ss"):
        return name[:-1]
    return name


class GetGemsClient(BaseMarketClient):
    """Client for the GetGems marketplace public API.

    Auth via ``Authorization: Bearer <token>`` header.
    Floor prices from ``GET /v1/gifts/collections`` (cursor-paginated).
    """

    market_name = MarketName.GETGEMS
    base_url = GETGEMS_BASE_URL

    def __init__(self, token: str) -> None:
        super().__init__(token=token)
        # Populated by fetch_floor_prices() — maps collection display name
        # (e.g. "Scared Cat") to its on-chain address. Used by
        # fetch_collection_attributes() to resolve a name into the address
        # the /attributes/{address} endpoint requires.
        self._name_to_address: dict[str, str] = {}

    def _auth_headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self._token}"} if self._token else {}

    async def authenticate(self) -> None:
        """No-op — the bearer token is static and pre-issued via env.

        The ``Authorization`` header is attached to every request by
        :meth:`_auth_headers`.
        """
        if not self._token:
            from core.exceptions import ConfigError

            raise ConfigError("GetGems: GETGEMS_API_TOKEN is not set")
        logger.debug("GetGems: using static bearer token")

    async def fetch_floor_prices(self) -> list[FloorPrice]:
        """Fetch floor prices for all gift collections.

        Walks the cursor-paginated ``GET /v1/gifts/collections/top?kind=all``
        endpoint until the cursor is empty (capped at ``_MAX_PAGES`` pages).
        Each item is ``{place, collection:{address, name, …}, floorPrice}``
        where ``floorPrice`` is the floor in TON (GetGems uses TON 1:1).

        As a side effect this also refreshes :attr:`_name_to_address`, which
        :meth:`fetch_collection_attributes` relies on to resolve a
        collection's display name to its on-chain address.
        """
        prices: list[FloorPrice] = []
        name_to_address: dict[str, str] = {}
        cursor: str | None = None
        pages = 0

        for page in range(_MAX_PAGES):
            params: dict[str, Any] = {"kind": "all"}
            if cursor:
                params["after"] = cursor

            data = await self._request(
                "GET", "/v1/gifts/collections/top", params=params,
            )

            # Response envelope: {"success": true, "response": {cursor, items}}
            response = data.get("response", data) if isinstance(data, dict) else {}
            items = response.get("items", []) if isinstance(response, dict) else []
            cursor = response.get("cursor") if isinstance(response, dict) else None
            pages = page + 1

            for item in items if isinstance(items, list) else []:
                # Item shape: {place, collection: {address, name, …}, floorPrice}
                collection = item.get("collection") or {}
                # Singularize so GetGems keys match the other markets'
                # ("Scared Cats" → "Scared Cat") — see helper docstring.
                name = _singularize_collection_name(collection.get("name") or "")
                address = collection.get("address")
                if not name:
                    continue
                if address:
                    name_to_address[name] = address
                floor = item.get("floorPrice")
                try:
                    price = float(floor) if floor is not None else None
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

            if not cursor:
                break

        if name_to_address:
            self._name_to_address = name_to_address

        logger.info(
            "GetGems: fetched %d floor prices (%d pages, %d addresses mapped)",
            len(prices), pages, len(name_to_address),
        )
        return prices

    async def fetch_collection_attributes(
        self, collection_name: str,
    ) -> dict[str, list[dict[str, Any]]]:
        """Fetch floor prices for models and backdrops of a collection.

        GetGems exposes per-trait floors via
        ``GET /v1/collection/attributes/{address}``, which returns
        ``{response: {attributes: [{traitType, values: [{value, count,
        minPrice, minPriceNano}]}]}}``. The endpoint keys off the on-chain
        collection *address*, so the name is resolved through
        :attr:`_name_to_address` (populated by :meth:`fetch_floor_prices`).

        Args:
            collection_name: Display name like "Scared Cat".

        Returns:
            ``{"models": [...], "backdrops": [...]}`` where each item is
            ``{"name", "price" (TON or None)}``. Empty lists if the
            collection name isn't mapped to an address on GetGems.
        """
        address = self._name_to_address.get(collection_name)
        if not address:
            # Name not in the cached map — try refreshing it once before
            # giving up. This covers the case where fetch_floor_prices()
            # hasn't run yet (e.g. attribute lookup at startup) or the
            # collection entered the top list after the last price cycle.
            await self._ensure_name_to_address()
            address = self._name_to_address.get(collection_name)
        if not address:
            logger.debug(
                "GetGems: no address mapped for %r — skipping attributes",
                collection_name,
            )
            return {"models": [], "backdrops": []}

        data = await self._request(
            "GET", f"/v1/collection/attributes/{address}",
        )
        response = data.get("response", data) if isinstance(data, dict) else {}
        attributes = (
            response.get("attributes", []) if isinstance(response, dict) else []
        )

        models: list[dict[str, Any]] = []
        backdrops: list[dict[str, Any]] = []
        for attr in attributes if isinstance(attributes, list) else []:
            trait = (attr.get("traitType") or "").lower()
            bucket = (
                models if trait == "model"
                else backdrops if trait == "backdrop"
                else None
            )
            if bucket is None:
                # GetGems also exposes "Symbol" — out of scope for the
                # {models, backdrops} contract shared with MRKT/Portal.
                continue
            for v in attr.get("values", []) if isinstance(attr.get("values"), list) else []:
                name = v.get("value")
                if not name:
                    continue
                min_price = v.get("minPrice")
                try:
                    price = float(min_price) if min_price is not None else None
                except (ValueError, TypeError):
                    price = None
                bucket.append({"name": name, "price": price})

        logger.info(
            "GetGems: %s — %d models, %d backdrops",
            collection_name, len(models), len(backdrops),
        )
        return {"models": models, "backdrops": backdrops}

    async def _ensure_name_to_address(self) -> None:
        """Populate :attr:`_name_to_address` if it's currently empty.

        Runs one pass over the top-collections cursor (the same walk
        :meth:`fetch_floor_prices` performs, but without building
        :class:`FloorPrice` entries). Cheap because the data is already
        what we'd request for prices anyway.
        """
        if self._name_to_address:
            return
        cursor: str | None = None
        for page in range(_MAX_PAGES):
            params: dict[str, Any] = {"kind": "all"}
            if cursor:
                params["after"] = cursor
            data = await self._request(
                "GET", "/v1/gifts/collections/top", params=params,
            )
            response = data.get("response", data) if isinstance(data, dict) else {}
            items = response.get("items", []) if isinstance(response, dict) else []
            cursor = response.get("cursor") if isinstance(response, dict) else None
            for item in items if isinstance(items, list) else []:
                collection = item.get("collection") or {}
                name = _singularize_collection_name(collection.get("name") or "")
                address = collection.get("address")
                if name and address:
                    self._name_to_address[name] = address
            if not cursor:
                break
