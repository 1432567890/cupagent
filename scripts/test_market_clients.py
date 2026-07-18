#!/usr/bin/env python3
"""Smoke test for Tonnel + XGift market clients.

Tests the actual auth + fetch flow against the real APIs using the
configured Kurigram session. Run from project root:

    python3 scripts/test_market_clients.py

Reports per-client: auth OK/FAIL, number of floor prices, sample prices.
"""

from __future__ import annotations

import asyncio
import logging
import sys
from pathlib import Path

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("test")


async def build_provider():
    """Build the Kurigram InitDataProvider from settings/session."""
    from config.settings import get_settings
    from user.session.store import SessionStore
    from services.init_data_provider import KurigramInitDataProvider

    settings = get_settings()
    store = SessionStore()
    if not (settings.SESSION_STRING or store.exists()):
        raise RuntimeError("No Kurigram session available")
    provider = KurigramInitDataProvider(
        api_id=settings.API_ID,
        api_hash=settings.API_HASH,
        session_string=settings.SESSION_STRING or None,
        lang_code="ru",
        system_lang_code="ru",
        session_store=store,
    )
    return provider


async def test_init_data(provider, bot_username: str, app_short_name: str) -> None:
    """Try to obtain initData for the given bot/app — print length."""
    try:
        init_data = await provider.get_init_data(bot_username, app_short_name)
        if init_data and "tgWebAppData" not in init_data and "user=" in init_data:
            print(f"  initData OK (len={len(init_data)})")
        elif init_data:
            print(f"  initData OK (len={len(init_data)}, starts={init_data[:60]!r})")
        else:
            print("  initData FAILED (empty)")
    except Exception as e:
        print(f"  initData FAILED: {type(e).__name__}: {e}")


async def test_tonnel(provider) -> None:
    print("\n=== Tonnel ===")

    from markets.tonnel.client import TonnelClient

    client = TonnelClient(provider, bot_username="Tonnel_Network_bot", app_short_name="tonnel")
    try:
        print("  authenticating (Tonnel_Network_bot/tonnel)...")
        await client.authenticate()
        print("  fetching floor prices (bulk filterStats)...")
        prices = await client.fetch_floor_prices()
        listed = [p for p in prices if p.price is not None]
        print(f"  floors: {len(prices)} total, {len(listed)} listed")
        for p in prices[:5]:
            print(f"    {p.gift_name}: {p.price}")
        print("  fetching attributes for 'Scared Cat'...")
        attrs = await client.fetch_collection_attributes("Scared Cat")
        print(f"    models: {len(attrs.get('models', []))}")
        for m in attrs.get("models", [])[:3]:
            print(f"      {m}")
    except Exception as e:
        print(f"  FAILED: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
    finally:
        await client.close()


async def test_xgift(provider) -> None:
    print("\n=== XGift ===")
    from markets.xgift.client import XGiftClient

    client = XGiftClient(provider, bot_username="xgift", app_short_name="app")
    try:
        print("  authenticating (xgift/app)...")
        await client.authenticate()
        print("  fetching floor prices...")
        prices = await client.fetch_floor_prices()
        listed = [p for p in prices if p.price is not None]
        print(f"  floors: {len(prices)} total, {len(listed)} listed")
        for p in prices[:5]:
            print(f"    {p.gift_name}: {p.price}")
        print("  fetching attributes for 'Scared Cat'...")
        attrs = await client.fetch_collection_attributes("Scared Cat")
        print(f"    models: {len(attrs.get('models', []))}, backdrops: {len(attrs.get('backdrops', []))}")
        for m in attrs.get("models", [])[:3]:
            print(f"      model: {m}")
        for b in attrs.get("backdrops", [])[:3]:
            print(f"      backdrop: {b}")
    except Exception as e:
        print(f"  FAILED: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
    finally:
        await client.close()


async def main() -> None:
    print("Building InitDataProvider...")
    provider = await build_provider()
    try:
        await test_tonnel(provider)
        await test_xgift(provider)
    finally:
        await provider.close()


if __name__ == "__main__":
    asyncio.run(main())
