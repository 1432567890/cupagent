"""MRKT Market client — https://api.tgmrkt.io

Authentication: POST /api/v1/auth with Telegram initData (from Kurigram).
Returns a UUID session token used for subsequent requests.
Floor prices: GET /api/v1/gifts/collections (floorPriceNanoTons field).

The MRKT API requires a ``Referer`` header pointing at its CDN origin
(``https://cdn.tgmrkt.io/``); requests without it are rejected with 403.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any

from core.constants import MRKT_BASE_URL, NANOTON
from core.types import FloorPrice, MarketName
from markets.base import BaseMarketClient
from services.init_data_provider import InitDataProvider

logger = logging.getLogger(__name__)

# MRKT rejects requests without this Referer with HTTP 403.
_MRKT_REFERER = "https://cdn.tgmrkt.io/"


class MrktClient(BaseMarketClient):
    """Client for the MRKT marketplace API.

    Auth via POST /api/v1/auth → receives a UUID token.
    Subsequent requests use ``Authorization: <token>`` header.
    Floor prices from GET /api/v1/gifts/collections (values in nanoTONs → GRAM).
    """

    market_name = MarketName.MRKT
    base_url = MRKT_BASE_URL

    def __init__(
        self,
        init_data_provider: InitDataProvider,
        bot_username: str = "mrkt",
        app_short_name: str = "app",
    ) -> None:
        # `token` arg of base class unused — we get initData dynamically
        super().__init__(token="")
        self._provider = init_data_provider
        self._bot_username = bot_username
        self._app_short_name = app_short_name
        self._session_token: str = ""

    @property
    def _default_headers(self) -> dict[str, str]:
        """Override to inject the Referer header MRKT requires."""
        return {
            "Accept": "application/json",
            "Content-Type": "application/json",
            "Referer": _MRKT_REFERER,
            "Origin": _MRKT_REFERER.rstrip("/"),
        }

    def _auth_headers(self) -> dict[str, str]:
        headers: dict[str, str] = {}
        if self._session_token:
            headers["Authorization"] = self._session_token
        return headers

    async def authenticate(self) -> None:
        """Authenticate and store the session token.

        Endpoint: POST /api/v1/auth
        Body: ``{"data": "<initData>"}`` — MRKT only needs the initData string.
        Response: ``{"token": "<uuid>", ...}``

        Self-healing: if the cached initData is rejected, drop it, fetch
        a fresh one and try once more — so a stale-but-not-expired cached
        value doesn't lock the client out until the cache TTL.
        """
        body: dict[str, Any] = {
            "data": await self._provider.get_init_data(
                self._bot_username, self._app_short_name
            )
        }
        try:
            data = await self._request(
                "POST", "/api/v1/auth", json_body=body, retry_on_401=False,
            )
        except Exception:
            await self._invalidate_auth()
            body = {
                "data": await self._provider.get_init_data(
                    self._bot_username, self._app_short_name
                )
            }
            data = await self._request(
                "POST", "/api/v1/auth", json_body=body, retry_on_401=False,
            )
        self._session_token = data.get("token", "") if isinstance(data, dict) else ""
        if not self._session_token:
            from core.exceptions import MarketAuthError

            raise MarketAuthError("MRKT: no session token in auth response")
        logger.info("MRKT: authenticated, token=%s…", self._session_token[:8])

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

        Values are in nanoTONs, converted to GRAM.
        """
        data = await self._request("GET", "/api/v1/gifts/collections")
        prices: list[FloorPrice] = []

        for item in data if isinstance(data, list) else []:
            name = item.get("name") or item.get("title")
            floor_nano = item.get("floorPriceNanoTons")
            if not name:
                continue
            price = floor_nano / NANOTON if floor_nano is not None else None
            prices.append(
                FloorPrice(
                    gift_name=name,
                    market=self.market_name,
                    price=price,
                    updated_at=datetime.now(timezone.utc),
                )
            )

        logger.info("MRKT: fetched %d floor prices", len(prices))
        return prices

    async def fetch_collection_attributes(
        self, collection_name: str,
    ) -> dict[str, list[dict[str, Any]]]:
        """Fetch floor prices for models and backdrops of a collection.

        MRKT exposes two POST endpoints that return per-attribute floors:
            POST /api/v1/gifts/models     body {"collections": ["Name"]}
            POST /api/v1/gifts/backdrops  body {"collections": ["Name"]}

        Note the inconsistent field naming on MRKT's side: models use
        ``floorPriceNanoTons`` while backdrops use ``floorNanoTons``.

        Args:
            collection_name: Display name like "Surge Board" or "Scared Cat".

        Returns:
            ``{"models": [...], "backdrops": [...]}`` where each item is
            ``{"name", "price" (GRAM or None)}``.
        """
        body = {"collections": [collection_name]}

        models_raw, backdrops_raw = await asyncio.gather(
            self._request("POST", "/api/v1/gifts/models", json_body=body),
            self._request("POST", "/api/v1/gifts/backdrops", json_body=body),
            return_exceptions=False,
        )

        models = []
        for item in models_raw if isinstance(models_raw, list) else []:
            name = item.get("modelName") or item.get("modelTitle")
            floor_nano = item.get("floorPriceNanoTons")
            if not name:
                continue
            models.append({
                "name": name,
                "price": floor_nano / NANOTON if floor_nano else None,
            })

        backdrops = []
        for item in backdrops_raw if isinstance(backdrops_raw, list) else []:
            name = item.get("backdropName")
            floor_nano = item.get("floorNanoTons")
            if not name:
                continue
            backdrops.append({
                "name": name,
                "price": floor_nano / NANOTON if floor_nano else None,
            })

        logger.info(
            "MRKT: %s — %d models, %d backdrops",
            collection_name, len(models), len(backdrops),
        )
        return {"models": models, "backdrops": backdrops}
