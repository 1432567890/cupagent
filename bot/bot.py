"""Telegram bot factory — aiogram 3.

Creates a Bot instance with default HTML parse mode,
registers routers and middlewares (whitelist, parse mode).
"""

from __future__ import annotations

import logging
from typing import Any

from aiogram import Bot, Dispatcher, Router
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode

from bot.handlers.start import router as start_router
from bot.handlers.prices import router as prices_router
from bot.handlers.chat import router as chat_router
from bot.handlers.guest_chat import router as guest_chat_router
from bot.middleware.duplicate_spam import DuplicateSpamMiddleware
from bot.middleware.rate_limit import RateLimitMiddleware
from bot.middleware.whitelist import WhitelistMiddleware
from bot.middlewares import DefaultParseModeMiddleware

logger = logging.getLogger(__name__)


def create_bot(
    token: str,
    *,
    whitelist: set[int] | None = None,
    redis: Any = None,
    rate_limit_max: int = 5,
    rate_limit_window: int = 15,
    antispam_duplicate_window: int = 60,
    antispam_duplicate_threshold: int = 3,
    antispam_block_seconds: int = 300,
) -> tuple[Bot, Dispatcher]:
    """Create and configure the aiogram Bot and Dispatcher.

    Args:
        token: Telegram bot token from @BotFather.
        whitelist: Optional set of allowed user IDs. None = open mode.
        redis: Optional Redis client for rate-limiter and anti-spam.
        rate_limit_max: Max free-text messages per user in the window.
        rate_limit_window: Sliding-window size in seconds.
        antispam_duplicate_window: Seconds to remember recent messages for
            duplicate detection.
        antispam_duplicate_threshold: How many identical messages before
            hard-block.
        antispam_block_seconds: Duration of the hard-block.

    Returns:
        Tuple of (Bot, Dispatcher) ready to run.
    """
    # DefaultBotProperties sets HTML parse_mode globally —
    # no need to specify parse_mode in every message call.
    bot = Bot(
        token=token,
        default=DefaultBotProperties(parse_mode=ParseMode.HTML),
    )

    dp = Dispatcher()

    # Register outer middlewares (run on every update)
    dp.update.outer_middleware(DefaultParseModeMiddleware(parse_mode="HTML"))

    # Build a messages router with whitelist + rate-limit + antispam gating
    messages_router = Router(name="messages")
    # aiogram 3: register message-level middleware
    messages_router.message.middleware(WhitelistMiddleware(whitelist))
    messages_router.message.middleware(
        RateLimitMiddleware(
            redis,
            max_messages=rate_limit_max,
            window_seconds=rate_limit_window,
        )
    )
    messages_router.message.middleware(
        DuplicateSpamMiddleware(
            redis,
            window_seconds=antispam_duplicate_window,
            threshold=antispam_duplicate_threshold,
            block_seconds=antispam_block_seconds,
        )
    )
    # Guest messages (Telegram Business / managed chats) also go through
    # the same whitelist + rate limit + antispam — the event object is
    # still a Message.
    messages_router.guest_message.middleware(WhitelistMiddleware(whitelist))
    messages_router.guest_message.middleware(
        RateLimitMiddleware(
            redis,
            max_messages=rate_limit_max,
            window_seconds=rate_limit_window,
        )
    )
    messages_router.guest_message.middleware(
        DuplicateSpamMiddleware(
            redis,
            window_seconds=antispam_duplicate_window,
            threshold=antispam_duplicate_threshold,
            block_seconds=antispam_block_seconds,
        )
    )

    # Wire handlers
    messages_router.include_router(start_router)
    messages_router.include_router(prices_router)
    messages_router.include_router(chat_router)
    messages_router.include_router(guest_chat_router)

    dp.include_router(messages_router)

    mode = "open" if whitelist is None else f"whitelist({len(whitelist)})"
    logger.info("Bot: created (mode=%s, parse_mode=HTML)", mode)
    return bot, dp
