"""Repository for tracking bot users (Telegram accounts that messaged the bot).

Stores ``user_id`` (Telegram ID, BIGINT), ``first_name``, ``username`` and
an ``updated_at`` timestamp. Records are upserted on every message so the
table always reflects the user's current display name.

Keeps the same style as ``FloorPriceRepo``: asyncpg directly, no ORM at
runtime, schema created lazily via ``CREATE TABLE IF NOT EXISTS``.
"""

from __future__ import annotations

import logging
from typing import Any

import asyncpg

logger = logging.getLogger(__name__)

_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS users (
    user_id     BIGINT PRIMARY KEY,
    first_name  TEXT,
    username    TEXT,
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
"""

# Idempotent upsert: on conflict update display fields + bump updated_at.
# Username may be NULL (not every Telegram user has one).
_UPSERT_SQL = """
INSERT INTO users (user_id, first_name, username, updated_at)
VALUES ($1, $2, $3, NOW())
ON CONFLICT (user_id) DO UPDATE
SET first_name = EXCLUDED.first_name,
    username   = EXCLUDED.username,
    updated_at = EXCLUDED.updated_at;
"""

_COUNT_SQL = "SELECT COUNT(*) FROM users;"


class UserRepo:
    """Repository for the ``users`` table.

    Args:
        pool: asyncpg connection pool shared with the rest of the app.
    """

    def __init__(self, pool: asyncpg.Pool) -> None:
        self._pool = pool

    async def init_db(self) -> None:
        """Create the ``users`` table if it does not exist."""
        async with self._pool.acquire() as conn:
            await conn.execute(_CREATE_TABLE_SQL)
        logger.info("DB: users table ready")

    async def upsert(
        self,
        user_id: int,
        first_name: str | None,
        username: str | None,
    ) -> None:
        """Insert or update a user record.

        Safe to call on every message — the upsert is idempotent and only
        refreshes ``first_name``/``username``/``updated_at``.

        Args:
            user_id: Telegram user ID.
            first_name: User's display first name (may be None).
            username: User's @username without the leading ``@`` (may be None).
        """
        async with self._pool.acquire() as conn:
            await conn.execute(_UPSERT_SQL, user_id, first_name, username)

    async def count(self) -> int:
        """Return the total number of tracked users."""
        async with self._pool.acquire() as conn:
            return await conn.fetchval(_COUNT_SQL)
