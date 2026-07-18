"""Rate-limit middleware — drops spam from a single user.

Uses a Redis-backed sliding-window counter per ``user_id``. When a user
exceeds ``max_messages`` within ``window_seconds``, subsequent free-text
messages are silently dropped (no LLM call, no reply) until the window
clears.

Only applies to free-text chat (the LLM path). Commands (/start, /prices)
and non-Message updates are passed through unchanged.

Design:
- Redis key: ``oclp:rate:{user_id}`` — a sorted set of recent
  message timestamps. We trim entries older than the window and count
  the remainder. The key is given a TTL of ``window_seconds`` so it
  expires automatically when the user stops messaging.
- In-memory fallback: if Redis is unavailable, we degrade gracefully to
  an in-process dict of deques (per-process only — not shared across
  workers, but better than nothing).
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import defaultdict, deque
from typing import Any, Awaitable, Callable

from aiogram import BaseMiddleware
from aiogram.types import Message, TelegramObject

logger = logging.getLogger(__name__)

_REDIS_KEY = "oclp:rate:{user_id}"


class RateLimitMiddleware(BaseMiddleware):
    """Sliding-window rate limiter, per user, backed by Redis.

    Args:
        redis: Optional ``redis.asyncio.Redis`` client. If None, an
            in-process dict is used (not shared across workers).
        max_messages: How many free-text messages a user may send
            within the window.
        window_seconds: Size of the sliding window in seconds.
    """

    def __init__(
        self,
        redis: Any = None,
        *,
        max_messages: int = 5,
        window_seconds: int = 15,
    ) -> None:
        self._redis = redis
        self._max = max_messages
        self._window = window_seconds
        # In-memory fallback: user_id -> deque of timestamps.
        self._local: dict[int, deque[float]] = defaultdict(deque)
        self._lock = asyncio.Lock()

    async def __call__(
        self,
        handler: Callable[[TelegramObject, dict[str, Any]], Awaitable[Any]],
        event: TelegramObject,
        data: dict[str, Any],
    ) -> Any:
        # Only gate free-text Messages — let commands and other updates
        # through unchanged.
        if not isinstance(event, Message):
            return await handler(event, data)

        text = (event.text or "").strip()
        if not text or text.startswith("/"):
            return await handler(event, data)

        user = event.from_user
        if user is None:
            return await handler(event, data)

        allowed = await self._check(user.id)
        if not allowed:
            logger.info(
                "rate_limit: dropped message from user %d (over %d/%ds)",
                user.id, self._max, self._window,
            )
            return None  # silently drop — no reply, no LLM call

        return await handler(event, data)

    async def _check(self, user_id: int) -> bool:
        """Return True if the user may send another message now."""
        if self._redis is not None:
            return await self._check_redis(user_id)
        return await self._check_local(user_id)

    async def _check_redis(self, user_id: int) -> bool:
        """Sliding window via a Redis sorted set of timestamps."""
        import json

        key = _REDIS_KEY.format(user_id=user_id)
        now = time.time()
        cutoff = now - self._window

        try:
            pipe = self._redis.pipeline()
            # 1. Drop entries older than the window.
            pipe.zremrangebyscore(key, 0, cutoff)
            # 2. Count remaining entries.
            pipe.zcard(key)
            # 3. Add the current timestamp.
            pipe.zadd(key, {str(now): now})
            # 4. Refresh TTL so the key expires when the user goes idle.
            pipe.expire(key, self._window)
            _, count, _, _ = await pipe.execute()
        except Exception:
            logger.warning(
                "rate_limit: Redis check failed, falling back to in-memory",
                exc_info=True,
            )
            return await self._check_local(user_id)

        # ``count`` is the size BEFORE adding the current message.
        # Allow if we haven't yet hit the cap.
        return count < self._max

    async def _check_local(self, user_id: int) -> bool:
        """In-process sliding window — fallback when Redis is unavailable."""
        now = time.time()
        cutoff = now - self._window
        async with self._lock:
            dq = self._local[user_id]
            while dq and dq[0] < cutoff:
                dq.popleft()
            if len(dq) >= self._max:
                return False
            dq.append(now)
            return True
