"""Anti-spam middleware — drops repeated identical messages.

Detects two spam patterns:

1. **Exact duplicate within a time window** — a user sends the same
   message they already sent in the last N seconds. The duplicate is
   silently dropped (no LLM call, no reply).

2. **Repeat threshold** — if a user sends the *same* message more than
   ``threshold`` times within the window, they are hard-blocked for
   ``block_seconds`` (even if the message changes afterwards).

State is kept in Redis (keyed per user). Falls back to in-process
storage if Redis is unavailable.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import time
from collections import defaultdict, deque
from typing import Any, Awaitable, Callable

from aiogram import BaseMiddleware
from aiogram.types import Message, TelegramObject

logger = logging.getLogger(__name__)

_REDIS_KEY = "cupagent:antispam:{user_id}"
_REDIS_BLOCK_KEY = "cupagent:antispam:block:{user_id}"


def _msg_hash(text: str) -> str:
    """Deterministic hash of a message for dedup comparison."""
    return hashlib.sha256(text.strip().lower().encode()).hexdigest()[:16]


class DuplicateSpamMiddleware(BaseMiddleware):
    """Drop identical repeated messages from the same user.

    Args:
        redis: Optional ``redis.asyncio.Redis`` client.
        window_seconds: How far back to look for duplicates.
        threshold: Max identical messages before hard-block.
        block_seconds: Duration of the hard-block.
    """

    def __init__(
        self,
        redis: Any = None,
        *,
        window_seconds: int = 60,
        threshold: int = 3,
        block_seconds: int = 300,
    ) -> None:
        self._redis = redis
        self._window = window_seconds
        self._threshold = threshold
        self._block = block_seconds
        # In-memory fallback: user_id -> deque of (timestamp, hash).
        self._local: dict[int, deque[tuple[float, str]]] = defaultdict(deque)
        self._blocked_until: dict[int, float] = {}
        self._lock = asyncio.Lock()

    async def __call__(
        self,
        handler: Callable[[TelegramObject, dict[str, Any]], Awaitable[Any]],
        event: TelegramObject,
        data: dict[str, Any],
    ) -> Any:
        if not isinstance(event, Message):
            return await handler(event, data)

        text = (event.text or "").strip()
        if not text or text.startswith("/"):
            return await handler(event, data)

        user = event.from_user
        if user is None:
            return await handler(event, data)

        verdict = await self._check(user.id, text)
        if verdict == "drop":
            logger.info(
                "antispam: dropped duplicate from user %d", user.id,
            )
            return None
        if verdict == "block":
            logger.warning(
                "antispam: hard-blocked user %d for %ds (spam threshold)",
                user.id, self._block,
            )
            return None

        return await handler(event, data)

    async def _check(self, user_id: int, text: str) -> str:
        """Return ``\"pass\"``, ``\"drop\"``, or ``\"block\"``."""
        if self._redis is not None:
            return await self._check_redis(user_id, text)
        return await self._check_local(user_id, text)

    async def _check_redis(self, user_id: int, text: str) -> str:
        """Redis-backed check: duplicate window + hard-block TTL."""
        try:
            # 1. Check hard-block.
            block_key = _REDIS_BLOCK_KEY.format(user_id=user_id)
            blocked = await self._redis.get(block_key)
            if blocked is not None:
                return "block"

            key = _REDIS_KEY.format(user_id=user_id)
            now = time.time()
            cutoff = now - self._window
            h = _msg_hash(text)

            pipe = self._redis.pipeline()
            # Drop old entries.
            pipe.zremrangebyscore(key, 0, cutoff)
            # Count entries with the SAME hash in the window.
            pipe.zcount(key, cutoff, now)
            # Add the new entry (score = timestamp, member = hash).
            pipe.zadd(key, {h: now})
            pipe.expire(key, self._window)
            _, count, _, _ = await pipe.execute()

            # ``count`` is BEFORE adding the current message.
            if count >= self._threshold:
                # Hard-block this user.
                await self._redis.set(
                    block_key, "1", ex=self._block,
                )
                return "block"

            if count > 0:
                # Same message already sent in the window — drop.
                return "drop"

            return "pass"

        except Exception:
            logger.warning(
                "antispam: Redis check failed, falling back to in-memory",
                exc_info=True,
            )
            return await self._check_local(user_id, text)

    async def _check_local(self, user_id: int, text: str) -> str:
        """In-process fallback when Redis is unavailable."""
        now = time.time()
        cutoff = now - self._window
        h = _msg_hash(text)
        async with self._lock:
            # Check hard-block.
            until = self._blocked_until.get(user_id, 0)
            if until > now:
                return "block"

            dq = self._local[user_id]
            while dq and dq[0][0] < cutoff:
                dq.popleft()

            same_count = sum(1 for _, mh in dq if mh == h)

            if same_count >= self._threshold:
                self._blocked_until[user_id] = now + self._block
                return "block"

            dq.append((now, h))

            if same_count > 0:
                return "drop"

        return "pass"
