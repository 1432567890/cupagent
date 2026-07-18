"""User-tracker middleware — persists every Telegram user that messages the bot.

For every incoming ``Message`` (regular or guest), upserts the sender's
``user_id``, ``first_name`` and ``username`` into the ``users`` table via
``UserRepo``. The upsert runs as a **fire-and-forget** background task so
it never blocks the middleware chain — even if PostgreSQL is slow or
unavailable, the user still gets a normal response.

Intentionally registered **first** in the message middleware chain so it
sees every message (including ones later dropped by whitelist/antispam).
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Awaitable, Callable

from aiogram import BaseMiddleware
from aiogram.types import Message, TelegramObject

logger = logging.getLogger(__name__)


class UserTrackerMiddleware(BaseMiddleware):
    """Persist every user that sends a message to the bot.

    Args:
        user_repo: ``UserRepo`` instance. If None, the middleware becomes
            a no-op (useful for tests / when DB is unavailable).
    """

    def __init__(self, user_repo: Any = None) -> None:
        self._repo = user_repo

    async def __call__(
        self,
        handler: Callable[[TelegramObject, dict[str, Any]], Awaitable[Any]],
        event: TelegramObject,
        data: dict[str, Any],
    ) -> Any:
        # Only Messages with a from_user can be tracked.
        if (
            self._repo is not None
            and isinstance(event, Message)
            and event.from_user is not None
        ):
            user = event.from_user
            # Fire-and-forget: never block the request on a DB write.
            # Errors are logged inside _safe_upsert, not propagated.
            asyncio.create_task(
                self._safe_upsert(
                    user_id=user.id,
                    first_name=user.first_name,
                    username=user.username,
                )
            )
        return await handler(event, data)

    async def _safe_upsert(
        self,
        *,
        user_id: int,
        first_name: str | None,
        username: str | None,
    ) -> None:
        """Upsert user record, swallowing and logging any DB error."""
        try:
            await self._repo.upsert(user_id, first_name, username)
        except Exception:
            logger.warning(
                "user_tracker: failed to upsert user %d",
                user_id,
                exc_info=True,
            )
