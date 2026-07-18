"""Grapes Market client — https://api.grapesmarket.xyz

Authentication: static API key via X-API-Token header.
Floor prices: GET /api/market/stats?type=collection
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from core.constants import GRAPES_BASE_URL
from core.types import FloorPrice, MarketName
from markets.base import BaseMarketClient

logger = logging.getLogger(__name__)


class GrapesClient(BaseMarketClient):
    """Client for the Grapes marketplace API.

    Auth via ``X-API-Token`` header containing a static API key.
    Floor prices come from ``GET /api/market/stats?type=collection``.
    """

    market_name = MarketName.GRAPES
    base_url = GRAPES_BASE_URL

    def _auth_headers(self) -> dict[str, str]:
        return {"X-API-Token": self._token}

    async def authenticate(self) -> None:
        """No-op for Grapes — API key is static, no login needed.

        The X-API-Token header is attached to every request via _auth_headers.
        """
        if not self._token:
            from core.exceptions import ConfigError

            raise ConfigError("Grapes: GRAPES_API_TOKEN is not set")
        logger.debug("Grapes: using static API key auth")

    async def fetch_floor_prices(self) -> list[FloorPrice]:
        """Fetch floor prices for all gift collections.

        Returns list of FloorPrice with price in TON (GRAM ≈ TON).
        Collections with no listings return price=None.
        """
        data = await self._request(
            "GET", "/api/market/stats", params={"type": "collection"}
        )
        items = data.get("items", []) if isinstance(data, dict) else data
        prices: list[FloorPrice] = []

        for item in items:
            name = item.get("name")
            floor = item.get("floor")
            if not name:
                continue
            prices.append(
                FloorPrice(
                    gift_name=name,
                    market=self.market_name,
                    price=float(floor) if floor is not None else None,
                    updated_at=datetime.now(timezone.utc),
                )
            )

        logger.info("Grapes: fetched %d floor prices", len(prices))
        return prices
