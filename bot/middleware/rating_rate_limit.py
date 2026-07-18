"""Rating-based rate-limit middleware — cooldowns tied to Telegram Star spending level.

Enforces a per-user cooldown between consecutive free-text (LLM) requests,
where the cooldown duration is determined by the user's Telegram Star rating
level (``ChatFullInfo.rating.level``). Higher-rated users get shorter cooldowns.

Tier mapping (from ``core.constants.RATING_RATE_LIMITS``):
    level 0   → 5 min
    level 1   → 1 min
    level 2–5 → 10 sec
    level 6+  → 1 sec

Only applies to free-text chat (the LLM path). Commands (/start, /prices,
/help, …) and non-Message updates are passed through unchanged.

Design:
- Rating level is fetched via ``event.bot.get_chat(user_id)`` and cached in
  Redis for ``RATING_LEVEL_CACHE_TTL`` seconds to avoid hitting the Bot API
  on every message.
- The last-allowed-request timestamp is stored in Redis under
  ``RATING_RATE_LIMIT_KEY``. On each free-text message, the middleware checks
  whether enough time has elapsed since the last request for the user's tier.
- Falls back to the most permissive tier (level 6+, 1 s) when Redis or the
  Bot API call fails — availability over strictness.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Awaitable, Callable

from aiogram import BaseMiddleware
from aiogram.types import ChatFullInfo, Message, TelegramObject

from core.constants import (
    RATING_LEVEL_CACHE_KEY,
    RATING_LEVEL_CACHE_TTL,
    RATING_RATE_LIMIT_KEY,
    RATING_RATE_LIMITS,
)

logger = logging.getLogger(__name__)

# Fallback: most permissive tier when we can't determine the rating.
_FALLBACK_COOLDOWN = 1.0


def _cooldown_for_level(level: int) -> float:
    """Return the cooldown in seconds for a given rating level.

    Iterates over ``RATING_RATE_LIMITS`` tuples
    ``(min_level, max_level, cooldown_seconds)`` and returns the matching
    cooldown. Falls back to ``_FALLBACK_COOLDOWN`` if no tier matches
    (should not happen with the default table).
    """
    for min_lvl, max_lvl, cooldown in RATING_RATE_LIMITS:
        if min_lvl <= level <= max_lvl:
            return cooldown
    return _FALLBACK_COOLDOWN


class RatingRateLimitMiddleware(BaseMiddleware):
    """Per-user cooldown gate driven by Telegram Star spending rating level.

    Args:
        redis: ``redis.asyncio.Redis`` client. If None, the middleware is
            effectively disabled (all messages pass through).

    The Bot instance is obtained from ``event.bot`` at call time.
    """

    def __init__(self, redis: Any = None) -> None:
        self._redis = redis

    # ── aiogram middleware entry point ──────────────────────────────────

    async def __call__(
        self,
        handler: Callable[[TelegramObject, dict[str, Any]], Awaitable[Any]],
        event: TelegramObject,
        data: dict[str, Any],
    ) -> Any:
        # Pass through non-Messages (CallbackQuery, etc.) unchanged.
        if not isinstance(event, Message):
            return await handler(event, data)

        # Commands always pass through.
        text = (event.text or "").strip()
        if not text or text.startswith("/"):
            return await handler(event, data)

        user = event.from_user
        if user is None:
            return await handler(event, data)

        # If no Redis — allow everything (fail-open).
        if self._redis is None:
            return await handler(event, data)

        bot = event.bot
        if bot is None:
            return await handler(event, data)

        allowed, cooldown = await self._check(user.id, bot)
        if not allowed:
            logger.info(
                "rating_rate_limit: dropped message from user %d "
                "(level cooldown=%.1fs)",
                user.id,
                cooldown,
            )
            return None  # silently drop

        return await handler(event, data)

    # ── internal logic ─────────────────────────────────────────────────

    async def _check(self, user_id: int, bot: Any) -> tuple[bool, float]:
        """Check whether the user may send a message right now.

        Returns:
            Tuple of (allowed, cooldown_seconds).
        """
        try:
            level = await self._get_rating_level(user_id, bot)
            cooldown = _cooldown_for_level(level)
            return await self._check_cooldown(user_id, cooldown), cooldown
        except Exception:
            logger.warning(
                "rating_rate_limit: check failed for user %d, allowing",
                user_id,
                exc_info=True,
            )
            return True, _FALLBACK_COOLDOWN

    async def _get_rating_level(self, user_id: int, bot: Any) -> int:
        """Resolve the user's rating level, using Redis cache when possible.

        Returns:
            Integer rating level (0+). Defaults to 0 if unknown.
        """
        cache_key = RATING_LEVEL_CACHE_KEY.format(user_id=user_id)

        try:
            # Try cache first.
            cached = await self._redis.get(cache_key)
            if cached is not None:
                return int(cached)

            # Cache miss — ask Telegram.
            chat_info: ChatFullInfo = await bot.get_chat(user_id)
            level = 0
            if chat_info.rating is not None:
                level = chat_info.rating.level

            # Persist in cache.
            await self._redis.set(
                cache_key,
                str(level),
                ex=RATING_LEVEL_CACHE_TTL,
            )
            return level

        except Exception:
            logger.warning(
                "rating_rate_limit: failed to fetch rating for user %d, "
                "defaulting to level 0",
                user_id,
                exc_info=True,
            )
            return 0

    async def _check_cooldown(self, user_id: int, cooldown: float) -> bool:
        """Return True if the cooldown has elapsed since the last request.

        Uses a simple Redis key storing the timestamp of the last allowed
        request. If the key is missing or expired, the user is allowed.

        A cooldown of ``0`` (or negative) means "no limit" — returns True
        immediately and skips the Redis round-trip entirely.
        """
        if cooldown <= 0:
            return True  # no limit configured for this tier

        key = RATING_RATE_LIMIT_KEY.format(user_id=user_id)
        now = time.time()

        try:
            last: str | None = await self._redis.get(key)
            if last is not None:
                elapsed = now - float(last)
                if elapsed < cooldown:
                    return False  # cooldown still active

            # Record this request and set TTL = cooldown so the key
            # auto-expires once the cooldown window has passed.
            await self._redis.set(key, str(now), ex=max(1, int(cooldown) + 1))
            return True

        except Exception:
            logger.warning(
                "rating_rate_limit: Redis cooldown check failed for user %d",
                user_id,
                exc_info=True,
            )
            return True  # fail-open
