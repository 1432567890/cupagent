"""Repository for floor price persistence using asyncpg directly.

Keeps it simple — no full ORM dependency at runtime.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Any

import asyncpg

from core.types import FloorPrice, MarketName

logger = logging.getLogger(__name__)

_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS floor_prices (
    id          SERIAL PRIMARY KEY,
    gift_name   VARCHAR(255) NOT NULL,
    market      VARCHAR(50)  NOT NULL,
    price       DOUBLE PRECISION,
    updated_at  TIMESTAMPTZ DEFAULT NOW(),
    CONSTRAINT floor_prices_name_market_uniq UNIQUE (gift_name, market)
);
"""

# Fallback: add the unique constraint if the table was created without it.
_ADD_CONSTRAINT_SQL = """
ALTER TABLE floor_prices
ADD CONSTRAINT floor_prices_name_market_uniq UNIQUE (gift_name, market);
"""

# Migrate legacy TIMESTAMP (without TZ) → TIMESTAMPTZ. asyncpg on Python 3.14
# refuses to coerce timezone-aware datetimes into a naive column, so we make
# the column tz-aware to match the aware datetimes produced by the clients.
_MIGRATE_TIMESTAMP_SQL = """
ALTER TABLE floor_prices
ALTER COLUMN updated_at TYPE TIMESTAMPTZ USING updated_at AT TIME ZONE 'UTC';
"""

_UPSERT_SQL = """
INSERT INTO floor_prices (gift_name, market, price, updated_at)
VALUES ($1, $2, $3, $4)
ON CONFLICT (gift_name, market) DO UPDATE
SET price       = EXCLUDED.price,
    updated_at  = EXCLUDED.updated_at;
"""


class FloorPriceRepo:
    """Repository for floor_price CRUD operations."""

    def __init__(self, pool: asyncpg.Pool) -> None:
        self._pool = pool

    async def init_db(self) -> None:
        """Create table, migrate legacy schema, and ensure constraints."""
        async with self._pool.acquire() as conn:
            await conn.execute(_CREATE_TABLE_SQL)
            # Migrate legacy TIMESTAMP column → TIMESTAMPTZ if needed.
            col = await conn.fetchval(
                "SELECT data_type FROM information_schema.columns "
                "WHERE table_name = 'floor_prices' AND column_name = 'updated_at'"
            )
            if col == "timestamp without time zone":
                logger.info("DB: migrating updated_at → TIMESTAMPTZ")
                await conn.execute(_MIGRATE_TIMESTAMP_SQL)
            try:
                await conn.execute(_ADD_CONSTRAINT_SQL)
            except asyncpg.DuplicateTableError:
                pass  # constraint already exists — table was created earlier
        logger.info("DB: floor_prices table ready")

    async def upsert(self, price: FloorPrice) -> None:
        """Insert or update a single floor price record."""
        async with self._pool.acquire() as conn:
            await conn.execute(
                _UPSERT_SQL,
                price.gift_name,
                price.market.value,
                price.price,
                price.updated_at,
            )

    async def upsert_many(self, prices: list[FloorPrice]) -> None:
        """Bulk insert/update floor prices using a transaction."""
        if not prices:
            return
        async with self._pool.acquire() as conn:
            async with conn.transaction():
                await conn.executemany(
                    _UPSERT_SQL,
                    [
                        (p.gift_name, p.market.value, p.price, p.updated_at)
                        for p in prices
                    ],
                )
        logger.info("DB: upserted %d floor prices", len(prices))

    async def get_by_market(
        self, market: MarketName
    ) -> list[FloorPrice]:
        """Fetch all floor prices for a specific market."""
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT gift_name, market, price, updated_at "
                "FROM floor_prices WHERE market = $1",
                market.value,
            )
        return [
            FloorPrice(
                gift_name=r["gift_name"],
                market=MarketName(r["market"]),
                price=r["price"],
                updated_at=r["updated_at"],
            )
            for r in rows
        ]

    async def get_by_market_and_gift(
        self, market: MarketName, gift_name: str
    ) -> FloorPrice | None:
        """Fetch a single floor price entry."""
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT gift_name, market, price, updated_at "
                "FROM floor_prices WHERE market = $1 AND gift_name = $2",
                market.value,
                gift_name,
            )
        if row is None:
            return None
        return FloorPrice(
            gift_name=row["gift_name"],
            market=MarketName(row["market"]),
            price=row["price"],
            updated_at=row["updated_at"],
        )
