#!/usr/bin/env python3
"""Full pipeline smoke test: PriceService → Tonnel + XGift.

Tests the real PriceService flow (authenticate + fetch_floor_prices +
fetch_collection_attributes) without DB/Redis — using in-memory fakes.
Run from project root:

    python3 scripts/test_price_pipeline.py
"""

from __future__ import annotations

import asyncio
import logging
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(
    level=logging.WARNING,  # quiet most logs
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
# But keep market client + price service INFO
logging.getLogger("markets").setLevel(logging.INFO)
logging.getLogger("services.price_service").setLevel(logging.INFO)


class FakeRepo:
    """In-memory FloorPriceRepo — no DB needed."""

    def __init__(self) -> None:
        self._store: list[Any] = []

    async def init_db(self) -> None:
        pass

    async def upsert_many(self, prices: list[Any]) -> None:
        self._store.extend(prices)

    async def get_by_market(self, market: Any) -> list[Any]:
        return [p for p in self._store if p.market == market]

    async def get_by_market_and_gift(self, market: Any, gift_name: str) -> Any:
        return next(
            (p for p in self._store if p.market == market and p.gift_name == gift_name),
            None,
        )


class FakeCache:
    """In-memory PriceCache — no Redis needed."""

    def __init__(self) -> None:
        self._store: dict[tuple, Any] = {}

    async def set_many(self, prices: list[Any]) -> None:
        for p in prices:
            self._store[(p.market, p.gift_name)] = p

    async def get_by_market(self, market: Any) -> list[Any]:
        return [p for (m, _), p in self._store.items() if m == market]

    async def get(self, market: Any, gift_name: str) -> Any:
        return self._store.get((market, gift_name))


async def main() -> None:
    from config.settings import get_settings
    from user.session.store import SessionStore
    from services.init_data_provider import KurigramInitDataProvider
    from services.price_service import PriceService

    settings = get_settings()
    store = SessionStore()
    if not (settings.SESSION_STRING or store.exists()):
        print("ERROR: no Kurigram session")
        return

    provider = KurigramInitDataProvider(
        api_id=settings.API_ID,
        api_hash=settings.API_HASH,
        session_string=settings.SESSION_STRING or None,
        lang_code="ru",
        system_lang_code="ru",
        session_store=store,
    )

    # Build PriceService with only Tonnel + XGift (skip Grapes/MRKT/Portal
    # to keep this smoke test focused + fast).
    ps = PriceService(
        grapes_api_key="",  # empty → Grapes will fail authenticate, fine
        init_data_provider=provider,
        cache=FakeCache(),
        repo=FakeRepo(),
        update_interval=99999,
    )

    # Drop the clients we don't want to exercise in this test.
    for m in list(ps._clients):
        if m.value not in ("tonnel", "xgift"):
            del ps._clients[m]

    print(f"\nRegistered clients: {list(ps._clients.keys())}")

    # 1. Update all markets (auth + fetch_floor_prices).
    print("\n=== _update_all_markets ===")
    await ps._update_all_markets()

    # 2. get_floor_prices per market.
    for name in ("tonnel", "xgift"):
        from core.types import MarketName
        m = MarketName(name)
        prices = await ps.get_floor_prices(m)
        listed = [p for p in prices if p.price is not None]
        print(f"\n{name}: {len(prices)} entries ({len(listed)} listed)")
        for p in prices[:3]:
            print(f"  {p.gift_name}: {p.price}")

    # 3. get_collection_attributes for a known collection.
    print("\n=== get_collection_attributes('Scared Cat') ===")
    attrs = await ps.get_collection_attributes("Scared Cat")
    for market, data in attrs.items():
        models = data.get("models", [])
        backdrops = data.get("backdrops", [])
        print(f"{market}: {len(models)} models, {len(backdrops)} backdrops")
        for mm in models[:2]:
            print(f"  model: {mm}")
        for bb in backdrops[:2]:
            print(f"  backdrop: {bb}")

    await ps.close()
    await provider.close()


if __name__ == "__main__":
    asyncio.run(main())
