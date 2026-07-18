"""Default parse mode middleware for aiogram 3.

Ensures the bot always uses HTML parse mode by default.
This is a redundant safety net — ``create_bot`` already sets
``DefaultBotProperties(parse_mode=HTML)`` on the Bot instance.
"""

from __future__ import annotations

from typing import Any, Awaitable, Callable

from aiogram import BaseMiddleware, Bot
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode
from aiogram.types import TelegramObject


class DefaultParseModeMiddleware(BaseMiddleware):
    """Sets default parse_mode on the bot so all messages use HTML."""

    def __init__(self, parse_mode: str = "HTML") -> None:
        self._parse_mode = parse_mode

    async def __call__(
        self,
        handler: Callable[[TelegramObject, dict[str, Any]], Awaitable[Any]],
        event: TelegramObject,
        data: dict[str, Any],
    ) -> Any:
        bot: Bot | None = data.get("bot")
        if bot is not None and bot.default is None:
            bot.default = DefaultBotProperties(
                parse_mode=ParseMode(self._parse_mode)
            )
        return await handler(event, data)
