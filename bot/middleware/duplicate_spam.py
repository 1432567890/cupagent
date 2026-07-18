"""Anti-spam middleware — drops only obvious spam floods.

The middleware targets one specific abuse pattern: a user sending the
SAME message many times in rapid succession (a classic bot/flood
attack). It does NOT rate-limit ordinary conversation — a user asking
the same question twice, or sending several different messages in a
row, is never blocked.

State is kept in Redis (keyed per user). Falls back to in-process
storage if Redis is unavailable.

Tuning (defaults are intentionally lenient):
    - ``window_seconds`` (60): how far back to count repeats of the
      same message.
    - ``threshold`` (10): how many identical messages within the window
      trigger a hard-block. Below this, messages always pass through.
    - ``block_seconds`` (120): duration of the hard-block once the
      threshold is crossed.
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

_REDIS_KEY = "cupagent:antispam:{user_id}:{hash}"
_REDIS_BLOCK_KEY = "cupagent:antispam:block:{user_id}"


def _msg_hash(text: str) -> str:
    """Deterministic hash of a message for dedup comparison."""
    return hashlib.sha256(text.strip().lower().encode()).hexdigest()[:16]


class DuplicateSpamMiddleware(BaseMiddleware):
    """Block only obvious spam floods (many identical messages in a row).

    Args:
        redis: Optional ``redis.asyncio.Redis`` client.
        window_seconds: How far back to count repeats of the same message.
        threshold: How many identical messages within the window trigger
            a hard-block. Below this, every message is passed through.
        block_seconds: Duration of the hard-block once triggered.
    """

    def __init__(
        self,
        redis: Any = None,
        *,
        window_seconds: int = 60,
        threshold: int = 10,
        block_seconds: int = 120,
    ) -> None:
        self._redis = redis
        self._window = window_seconds
        self._threshold = threshold
        self._block = block_seconds
        # In-memory fallback: (user_id, hash) -> deque of timestamps.
        self._local: dict[tuple[int, str], deque[float]] = defaultdict(deque)
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
        if verdict == "block":
            logger.warning(
                "antispam: blocked user %d for %ds (spam flood: %d+ identical "
                "messages in %ds)",
                user.id, self._block, self._threshold, self._window,
            )
            return None

        # "pass" — always forward to the handler. We intentionally do NOT
        # drop individual duplicates: a user legitimately re-asking the
        # same question (e.g. the bot's first answer was off) must always
        # get a response. Only the flood threshold above triggers a block.
        return await handler(event, data)

    async def _check(self, user_id: int, text: str) -> str:
        """Return ``\"pass\"`` or ``\"block\"``."""
        if self._redis is not None:
            return await self._check_redis(user_id, text)
        return await self._check_local(user_id, text)

    async def _check_redis(self, user_id: int, text: str) -> str:
        """Redis-backed check: count repeats of THIS message only."""
        try:
            # 1. Hard-block still active?
            block_key = _REDIS_BLOCK_KEY.format(user_id=user_id)
            if await self._redis.get(block_key) is not None:
                return "block"

            # 2. Count how many times THIS EXACT message was seen recently.
            # Key is per (user_id, hash) so different messages never
            # inflate each other's count. The previous implementation used
            # a single key per user with zcount over all hashes — that
            # turned the middleware into a generic rate-limiter (3 *any*
            # messages = block), which broke legitimate users.
            h = _msg_hash(text)
            key = _REDIS_KEY.format(user_id=user_id, hash=h)
            now = time.time()
            cutoff = now - self._window

            pipe = self._redis.pipeline()
            pipe.zremrangebyscore(key, 0, cutoff)
            pipe.zcount(key, cutoff, now)
            pipe.zadd(key, {h: now})
            pipe.expire(key, self._window)
            _, count_before_add, _, _ = await pipe.execute()

            # count_before_add is the number of times this same message
            # was already seen in the window (excluding the current one).
            if count_before_add >= self._threshold:
                await self._redis.set(block_key, "1", ex=self._block)
                return "block"

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
            until = self._blocked_until.get(user_id, 0)
            if until > now:
                return "block"

            # Per-(user, hash) deque — only counts repeats of this message.
            dq = self._local[(user_id, h)]
            while dq and dq[0] < cutoff:
                dq.popleft()

            if len(dq) >= self._threshold:
                self._blocked_until[user_id] = now + self._block
                return "block"

            dq.append(now)

        return "pass"
