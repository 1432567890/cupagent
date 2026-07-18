"""XGift Market client — https://app-api.xgift.tg

Authentication: Telegram ``initData`` is exchanged for a JWT via
``POST /auth/login`` (same Kurigram initData flow as MRKT/Portal). The
returned ``access_token`` is sent as ``Authorization: Bearer <jwt>`` on
subsequent requests.

Endpoints:
    POST /auth/login            — body ``{"dataType": "webapp",
                                    "data": "<initData>", "referredBy": null}``
                                    → ``{"access_token": "<jwt>", ...}``
    GET  /collections/          — ``?page=1&limit=200`` (no collectionType).
                                  Returns BOTH upgradable and base
                                  (non-upgradable) gifts. Upgradable gifts
                                  carry their name in ``giftName``; base
                                  gifts have ``giftName: null`` and expose
                                  their name in ``customGiftName``.
                                  → ``{gifts: [{giftName, customGiftName,
                                  giftNameFormatted, floorPrice, ...}]}``
    GET  /gifts/filters/<slug>  — ``{giftModel: [{model, floorPriceTon}],
                                  giftBackdrop: [{backdrop, floorPriceTon}]}``
                                  — model + backdrop floors for one
                                  collection, keyed by ``giftNameFormatted``
                                  slug. Base gifts have no slug/attributes.

Prices are in TON (1:1 with GRAM — XGift labels them "TON" but the value
is the same GRAM-denominated price used by every other market).
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from core.constants import XGIFT_BASE_URL
from core.types import FloorPrice, MarketName
from markets.base import BaseMarketClient
from services.init_data_provider import InitDataProvider

logger = logging.getLogger(__name__)


class XGiftClient(BaseMarketClient):
    """Client for the XGift marketplace API.

    Auth via ``POST /auth/login`` (initData → JWT). Subsequent requests
    use ``Authorization: Bearer <jwt>``. Floor prices from
    ``GET /collections/``; model + backdrop floors from
    ``GET /gifts/filters/<slug>``.
    """

    market_name = MarketName.XGIFT
    base_url = XGIFT_BASE_URL

    def __init__(
        self,
        init_data_provider: InitDataProvider,
        bot_username: str = "xgift",
        app_short_name: str = "app",
    ) -> None:
        super().__init__(token="")
        self._provider = init_data_provider
        self._bot_username = bot_username
        self._app_short_name = app_short_name
        self._jwt: str = ""
        # Populated by fetch_floor_prices() — maps collection display name
        # (e.g. "Scared Cat") to its XGift slug (giftNameFormatted, e.g.
        # "ScaredCat"). Used by fetch_collection_attributes() to resolve
        # a name into the /gifts/filters/<slug> path.
        self._name_to_slug: dict[str, str] = {}

    @property
    def _default_headers(self) -> dict[str, str]:
        """Override to inject Origin/Referer/UA XGift requires.

        XGift's API (app-api.xgift.tg) returns HTTP 403 for requests
        missing the browser-style ``Origin``/``Referer`` headers pointing
        at its web frontend (xgift.tg). We mirror exactly what the
        captured HAR traffic sends so the server treats us as the legit
        mini-app client.
        """
        return {
            "Accept": "application/json, text/plain, */*",
            "Content-Type": "application/json",
            "Origin": "https://xgift.tg",
            "Referer": "https://xgift.tg/",
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/605.1.15 (KHTML, like Gecko)"
            ),
        }

    def _auth_headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self._jwt}"} if self._jwt else {}

    async def authenticate(self) -> None:
        """Exchange initData for a JWT and cache it.

        Endpoint: ``POST /auth/login``
        Body: ``{"dataType": "webapp", "data": "<initData>", "referredBy": null}``
        Response: ``{"access_token": "<jwt>", ...}``

        Self-healing: if the cached initData is rejected (401 on login
        itself), drop it, fetch a fresh one and try once more.
        """
        body = await self._login_body()
        try:
            data = await self._request(
                "POST", "/auth/login", json_body=body, retry_on_401=False,
            )
        except Exception:
            await self._invalidate_auth()
            body = await self._login_body()
            data = await self._request(
                "POST", "/auth/login", json_body=body, retry_on_401=False,
            )
        token = data.get("access_token", "") if isinstance(data, dict) else ""
        if not token:
            from core.exceptions import MarketAuthError

            raise MarketAuthError("XGift: no access_token in login response")
        self._jwt = token
        logger.info("XGift: authenticated, jwt=%s…", self._jwt[:8])

    async def _login_body(self) -> dict[str, Any]:
        """Build the /auth/login body using fresh initData.

        ``dataType`` must be ``"webapp"`` (NOT ``"web"``) — the XGift
        backend rejects the latter. ``data`` is the raw Telegram WebApp
        initData string.
        """
        return {
            "dataType": "webapp",
            "data": await self._provider.get_init_data(
                self._bot_username, self._app_short_name,
            ),
            "referredBy": None,
        }

    async def _invalidate_auth(self) -> None:
        """Drop cached initData before a 401-driven re-auth.

        Called by :meth:`BaseMarketClient._request` when an API call
        returns 401 and the client is about to call :meth:`authenticate`
        again.
        """
        self._jwt = ""
        if hasattr(self._provider, "invalidate_cache"):
            await self._provider.invalidate_cache(
                self._bot_username, self._app_short_name,
            )

    async def fetch_floor_prices(self) -> list[FloorPrice]:
        """Fetch floor prices for all gift collections.

        ``GET /collections/?page=1&limit=200`` (no ``collectionType``)
        returns both upgradable and non-upgradable (base) gifts in one
        response — ``{gifts: [{giftName, customGiftName, giftNameFormatted,
        floorPrice, ...}]}``.

        Name resolution: upgradable gifts carry their display name in
        ``giftName``; non-upgradable (base) gifts have ``giftName: null``
        and expose their name in ``customGiftName`` instead. We fall back
        to ``customGiftName`` so base gifts (REDO, Coffin, etc.) are
        returned with their proper name.

        ``floorPrice`` is a decimal (TON == GRAM 1:1), int or string.

        As a side effect this refreshes :attr:`_name_to_slug`, which
        :meth:`fetch_collection_attributes` relies on to resolve a
        collection's display name to its XGift slug.
        """
        data = await self._request(
            "GET", "/collections/",
            params={"page": "1", "limit": "200"},
        )
        items = data.get("gifts", []) if isinstance(data, dict) else []

        prices: list[FloorPrice] = []
        name_to_slug: dict[str, str] = {}
        for item in items if isinstance(items, list) else []:
            # giftName for upgradable, customGiftName for base/unupgradable
            name = item.get("giftName") or item.get("customGiftName")
            if not name:
                continue
            slug = item.get("giftNameFormatted")
            if slug:
                name_to_slug[name] = slug
            price = _parse_price(item.get("floorPrice"))
            prices.append(
                FloorPrice(
                    gift_name=name,
                    market=self.market_name,
                    price=price,
                    updated_at=datetime.now(timezone.utc),
                )
            )

        if name_to_slug:
            self._name_to_slug = name_to_slug

        logger.info("XGift: fetched %d floor prices", len(prices))
        return prices

    async def _ensure_name_to_slug(self) -> None:
        """Populate :attr:`_name_to_slug` if it's currently empty.

        Runs one pass over /collections/ (the same call
        :meth:`fetch_floor_prices` performs, but only for the slug map).
        """
        if self._name_to_slug:
            return
        data = await self._request(
            "GET", "/collections/",
            params={"page": "1", "limit": "200"},
        )
        items = data.get("gifts", []) if isinstance(data, dict) else []
        for item in items if isinstance(items, list) else []:
            # Same fallback as fetch_floor_prices — base gifts have their
            # name in customGiftName, not giftName.
            name = item.get("giftName") or item.get("customGiftName")
            slug = item.get("giftNameFormatted")
            if name and slug:
                self._name_to_slug[name] = slug

    async def fetch_collection_attributes(
        self, collection_name: str,
    ) -> dict[str, list[dict[str, Any]]]:
        """Fetch floor prices for models and backdrops of a collection.

        ``GET /gifts/filters/<slug>`` returns
        ``{giftModel: [{model, floorPriceTon}], giftBackdrop: [{backdrop,
        floorPriceTon}]}``. The endpoint keys off the XGift slug
        (``giftNameFormatted``, e.g. "ScaredCat"), so the display name is
        resolved through :attr:`_name_to_slug` (populated by
        :meth:`fetch_floor_prices`).

        Args:
            collection_name: Display name like "Scared Cat".

        Returns:
            ``{"models": [...], "backdrops": [...]}`` where each item is
            ``{"name", "price" (GRAM or None)}``. Sorted by price asc.
        """
        slug = self._name_to_slug.get(collection_name)
        if not slug:
            # Name not in the cached map — try refreshing it once before
            # giving up. Covers the case where fetch_floor_prices() hasn't
            # run yet or the collection entered the list after the last
            # cycle.
            await self._ensure_name_to_slug()
            slug = self._name_to_slug.get(collection_name)
        if not slug:
            logger.debug(
                "XGift: no slug mapped for %r — skipping attributes",
                collection_name,
            )
            return {"models": [], "backdrops": []}

        data = await self._request("GET", f"/gifts/filters/{slug}")

        models = self._extract_attribute_list(data, "giftModel", "model")
        backdrops = self._extract_attribute_list(
            data, "giftBackdrop", "backdrop",
        )

        logger.info(
            "XGift: %s — %d models, %d backdrops",
            collection_name, len(models), len(backdrops),
        )
        return {"models": models, "backdrops": backdrops}

    @staticmethod
    def _extract_attribute_list(
        data: Any, key: str, name_field: str,
    ) -> list[dict[str, Any]]:
        """Pull an attribute floor list out of a /gifts/filters response.

        Each entry has a name field (``model`` or ``backdrop``) and
        ``floorPriceTon`` (decimal TON, string or number). Result is
        sorted by price ascending.
        """
        items = data.get(key, []) if isinstance(data, dict) else []
        out: list[dict[str, Any]] = []
        for item in items if isinstance(items, list) else []:
            name = item.get(name_field)
            if not name:
                continue
            out.append({"name": name, "price": _parse_price(item.get("floorPriceTon"))})
        out.sort(key=lambda m: (m["price"] is None, m["price"] or 0))
        return out


def _parse_price(raw: Any) -> float | None:
    """Coerce an XGift price (string/int/float) to a GRAM float, or None."""
    if raw is None:
        return None
    try:
        return float(raw)
    except (ValueError, TypeError):
        return None
