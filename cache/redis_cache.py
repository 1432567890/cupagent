"""Redis cache for floor prices.

Key pattern: oclp:floor_price:{market}:{gift_name}
Value: JSON {"price": float|null, "updated_at": ISO string}
TTL: 10 minutes (double the update interval for safety margin).

NOTE: The Redis client must be created with ``decode_responses=True``
so all keys and values are returned as ``str``, not ``bytes``.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime

import redis.asyncio as aioredis

from core.constants import REDIS_KEY_PREFIX
from core.types import FloorPrice, MarketName

logger = logging.getLogger(__name__)

_CACHE_TTL = 600  # seconds — 10 min


def _cache_key(market: MarketName, gift_name: str) -> str:
    return f"{REDIS_KEY_PREFIX}:{market.value}:{gift_name}"


def _serialize(price: FloorPrice) -> str:
    return json.dumps(
        {
            "price": price.price,
            "updated_at": price.updated_at.isoformat(),
        }
    )


def _deserialize(key: str, value: str) -> FloorPrice:
    data = json.loads(value)
    parts = key.split(":")
    # key format: oclp:floor_price:{market}:{gift_name}
    market = MarketName(parts[2])
    gift_name = ":".join(parts[3:])  # handle names with colons
    return FloorPrice(
        gift_name=gift_name,
        market=market,
        price=data.get("price"),
        updated_at=datetime.fromisoformat(data["updated_at"]),
    )


class PriceCache:
    """Redis-backed cache for floor prices."""

    def __init__(self, redis: aioredis.Redis) -> None:
        self._redis = redis

    async def set_many(self, prices: list[FloorPrice]) -> None:
        """Store multiple floor prices in cache."""
        if not prices:
            return
        pipe = self._redis.pipeline()
        for p in prices:
            key = _cache_key(p.market, p.gift_name)
            pipe.set(key, _serialize(p), ex=_CACHE_TTL)
        await pipe.execute()
        logger.debug("Cache: stored %d floor prices", len(prices))

    async def get_by_market(self, market: MarketName) -> list[FloorPrice]:
        """Fetch all cached prices for a market by pattern scan."""
        pattern = f"{REDIS_KEY_PREFIX}:{market.value}:*"
        keys: list[str] = []
        async for key in self._redis.scan_iter(match=pattern, count=200):
            # scan_iter returns str when decode_responses=True
            keys.append(key)
        if not keys:
            return []
        values = await self._redis.mget(keys)
        result = []
        for key, value in zip(keys, values):
            if value is None:
                continue
            result.append(_deserialize(key, value))
        return result

    async def get(
        self, market: MarketName, gift_name: str
    ) -> FloorPrice | None:
        """Fetch a single cached floor price."""
        key = _cache_key(market, gift_name)
        value = await self._redis.get(key)
        if value is None:
            return None
        return _deserialize(key, value)

    async def invalidate_market(self, market: MarketName) -> None:
        """Remove all cached prices for a market."""
        pattern = f"{REDIS_KEY_PREFIX}:{market.value}:*"
        keys: list[str] = []
        async for key in self._redis.scan_iter(match=pattern, count=200):
            keys.append(key)
        if keys:
            await self._redis.delete(*keys)
            logger.debug("Cache: invalidated %d keys for %s", len(keys), market)
