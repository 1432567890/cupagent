"""Whitelist middleware — restricts LLM access to approved users.

Behaviour:
- If WHITELIST env is empty → bot is open, everyone allowed (whitelist=None).
- If WHITELIST is set → only listed user IDs can chat with the LLM.
  Others get a polite "access denied" reply for free-text messages,
  but /start and /prices commands still work for everyone.
"""

from __future__ import annotations

import logging
import uuid
from typing import Any, Awaitable, Callable

from aiogram import BaseMiddleware
from aiogram.enums import ParseMode
from aiogram.types import (
    InlineQueryResultArticle,
    InputTextMessageContent,
    Message,
    TelegramObject,
)

logger = logging.getLogger(__name__)

# Commands that bypass the whitelist (always available)
_PUBLIC_COMMANDS = frozenset({"/start", "/prices", "/help"})

_DENY_TEXT = "<i>Бот сейчас на тех. перерыве. Возвращайтесь позже!</i>"


async def _deny(event: Message) -> None:
    """Send the access-denied notice via the channel matching the event.

    Guest messages (carry ``guest_query_id``) must be answered with
    ``answerGuestQuery``; regular messages use ``answer``.
    """
    bot = event.bot
    if event.guest_query_id is not None and bot is not None:
        await bot.answer_guest_query(
            guest_query_id=event.guest_query_id,
            result=InlineQueryResultArticle(
                id=str(uuid.uuid4()),
                title="Тех. перерыв",
                input_message_content=InputTextMessageContent(
                    message_text=_DENY_TEXT,
                    parse_mode=ParseMode.HTML,
                    link_preview_options={"is_disabled": True},
                ),
            ),
        )
        return
    await event.answer(
        _DENY_TEXT,
        link_preview_options={"is_disabled": True},
    )


class WhitelistMiddleware(BaseMiddleware):
    """Checks whether the current user is allowed to use the LLM.

    Sets ``data["whitelisted"]`` to True/False. Handlers can branch on it.
    If a non-whitelisted user sends free text, we reply with a notice.
    """

    def __init__(self, whitelist: set[int] | None) -> None:
        # None = open mode (no whitelist set)
        self._whitelist = whitelist

    async def __call__(
        self,
        handler: Callable[[TelegramObject, dict[str, Any]], Awaitable[Any]],
        event: TelegramObject,
        data: dict[str, Any],
    ) -> Any:
        # Only messages have a from_user to check
        if not isinstance(event, Message):
            return await handler(event, data)

        user = event.from_user
        user_id = user.id if user else None

        whitelisted = self._check(user_id)
        data["whitelisted"] = whitelisted
        data["open_mode"] = self._whitelist is None

        # If user is allowed, proceed normally
        if whitelisted:
            return await handler(event, data)

        # Non-whitelisted: allow public commands, block free text to LLM
        text = (event.text or "").strip()
        is_command = text.startswith("/")
        if is_command:
            return await handler(event, data)

        # Free text from non-whitelisted user → deny LLM access
        if user:
            try:
                await _deny(event)
            except Exception:
                logger.warning("whitelist: failed to send deny notice", exc_info=True)
        return None

    def _check(self, user_id: int | None) -> bool:
        """Return True if the user is allowed to use LLM features."""
        if self._whitelist is None:
            # Open mode — everyone allowed
            return True
        if user_id is None:
            return False
        return user_id in self._whitelist
