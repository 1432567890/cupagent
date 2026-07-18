"""Price service — orchestrates floor price fetching, caching, and DB storage.

Periodically fetches prices from all markets, stores in DB, caches in Redis.
Provides a method to get floor prices for a market (cache-first, DB fallback).
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from cache.redis_cache import PriceCache
from core.types import FloorPrice, MarketName
from db.repo import FloorPriceRepo
from markets.base import BaseMarketClient
from markets.grapesmp.client import GrapesClient
from markets.mrktmp.client import MrktClient
from markets.portalsmp.client import PortalClient
from markets.tonnel.client import TonnelClient
from markets.xgift.client import XGiftClient
from services.init_data_provider import InitDataProvider

logger = logging.getLogger(__name__)


class PriceService:
    """Orchestrates price updates and lookups across all markets."""

    def __init__(
        self,
        *,
        grapes_api_key: str,
        init_data_provider: InitDataProvider,
        cache: PriceCache,
        repo: FloorPriceRepo,
        update_interval: int = 300,
        grapes_bot_username: str = "grapesmarket_bot",
        mrkt_bot_username: str = "mrkt",
        portal_bot_username: str = "portals",
        tonnel_bot_username: str = "Tonnel_Network_bot",
        xgift_bot_username: str = "xgift",
        mrkt_app_short_name: str = "app",
        portal_app_short_name: str = "market",
        tonnel_app_short_name: str = "tonnel",
        xgift_app_short_name: str = "app",
        getgems_api_token: str = "",
    ) -> None:
        self._clients: dict[MarketName, BaseMarketClient] = {
            MarketName.GRAPES: GrapesClient(grapes_api_key),
            MarketName.MRKT: MrktClient(
                init_data_provider,
                bot_username=mrkt_bot_username,
                app_short_name=mrkt_app_short_name,
            ),
            MarketName.PORTAL: PortalClient(
                init_data_provider,
                bot_username=portal_bot_username,
                app_short_name=portal_app_short_name,
            ),
            MarketName.TONNEL: TonnelClient(
                init_data_provider,
                bot_username=tonnel_bot_username,
                app_short_name=tonnel_app_short_name,
            ),
            MarketName.XGIFT: XGiftClient(
                init_data_provider,
                bot_username=xgift_bot_username,
                app_short_name=xgift_app_short_name,
            ),
        }
        # GetGems is optional — only registered when a bearer token is set.
        if getgems_api_token:
            from markets.getgems.client import GetGemsClient

            self._clients[MarketName.GETGEMS] = GetGemsClient(getgems_api_token)
        self._cache = cache
        self._repo = repo
        self._interval = update_interval
        self._task: asyncio.Task[None] | None = None
        self._running = False

    async def start_periodic_updates(self) -> None:
        """Start the background task that fetches prices on a timer."""
        if self._running:
            return
        self._running = True
        await self._update_all_markets()
        self._task = asyncio.create_task(self._periodic_loop())
        logger.info(
            "PriceService: started periodic updates (interval=%ds)",
            self._interval,
        )

    async def stop_periodic_updates(self) -> None:
        """Cancel the background update task."""
        self._running = False
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("PriceService: stopped periodic updates")

    async def _periodic_loop(self) -> None:
        """Loop that sleeps and updates prices."""
        while self._running:
            await asyncio.sleep(self._interval)
            try:
                await self._update_all_markets()
            except Exception:
                logger.exception("PriceService: error in periodic update")

    async def _update_all_markets(self) -> None:
        """Fetch floor prices from all markets, store in DB and cache."""
        all_prices: list[FloorPrice] = []
        for market, client in self._clients.items():
            try:
                await client.authenticate()
                prices = await client.fetch_floor_prices()
                all_prices.extend(prices)
                logger.info(
                    "PriceService: %s — %d prices fetched", market.value, len(prices)
                )
            except Exception:
                logger.exception(
                    "PriceService: FAILED to fetch %s prices (auth or network error)",
                    market.value,
                )

        if all_prices:
            await self._repo.upsert_many(all_prices)
            await self._cache.set_many(all_prices)
            logger.info(
                "PriceService: updated %d total prices", len(all_prices)
            )

    async def get_floor_prices(
        self, market: MarketName
    ) -> list[FloorPrice]:
        """Get floor prices for a market — cache first, DB fallback."""
        cached = await self._cache.get_by_market(market)
        if cached:
            logger.debug(
                "PriceService: cache hit for %s (%d entries)",
                market,
                len(cached),
            )
            return sorted(cached, key=lambda p: p.gift_name)

        logger.debug("PriceService: cache miss for %s, hitting DB", market)
        db_prices = await self._repo.get_by_market(market)
        if db_prices:
            await self._cache.set_many(db_prices)
            return sorted(db_prices, key=lambda p: p.gift_name)

        logger.warning("PriceService: no prices found for %s anywhere", market)
        return []

    async def get_floor_price(
        self, market: MarketName, gift_name: str
    ) -> FloorPrice | None:
        """Get a single floor price — cache first, DB fallback."""
        cached = await self._cache.get(market, gift_name)
        if cached:
            return cached

        db_price = await self._repo.get_by_market_and_gift(market, gift_name)
        if db_price:
            await self._cache.set_many([db_price])
        return db_price

    async def get_collection_attributes(
        self, collection_name: str,
    ) -> dict[str, dict[str, list[dict[str, Any]]]]:
        """Fetch model + backdrop floor prices for a collection from all markets.

        Calls ``fetch_collection_attributes`` on each market client that
        supports it (MRKT, Portal). Markets that don't support it
        (Grapes, GetGems) are silently omitted from the result.

        Args:
            collection_name: Display name like "Scared Cat".

        Returns:
            ``{"mrkt": {"models": [...], "backdrops": [...]},
            "portal": {...}}`` — only markets that returned data.
        """
        out: dict[str, dict[str, list[dict[str, Any]]]] = {}
        for market, client in self._clients.items():
            fetcher = getattr(client, "fetch_collection_attributes", None)
            if fetcher is None:
                continue
            try:
                result = await fetcher(collection_name)
            except Exception:
                logger.warning(
                    "PriceService: attribute fetch failed for %s on %s",
                    collection_name, market.value, exc_info=True,
                )
                continue
            if result.get("models") or result.get("backdrops"):
                out[market.value] = result
        return out

    async def close(self) -> None:
        """Close all market client HTTP sessions."""
        await self.stop_periodic_updates()
        for client in self._clients.values():
            await client.close()
