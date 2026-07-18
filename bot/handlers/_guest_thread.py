"""Guest thread context — reply-chain history for Telegram Business guests.

Guest messages arrive via ``update.guest_message``. Telegram flattens
``reply_to_message`` to one level, so we maintain a Redis index
``message_id → thread_id`` to reconstruct reply chains.

Thread context:
  - Max ``_MAX_THREAD_MESSAGES`` messages per thread (tail-trimmed).
  - TTL ``_THREAD_TTL`` (3 days).
  - Per-thread key holds a JSON list of ``{role, content}`` dicts.
  - Per-message index key maps ``{chat_id}:{message_id} → thread_id``.

Resolution on incoming guest message:
  1. If ``reply_to_message`` is present, look up its ``message_id`` in the
     index. If found → continue that thread.
  2. Otherwise → create a new thread_id (uuid4).

Bot replies sent via ``answerGuestQuery`` get an ``inline_message_id``
(not a numeric ``message_id``), so we can't pre-index them. Instead, when
the user replies to a bot message, the incoming ``reply_to_message.message_id``
won't be in the index → we fall back to chat-scoped continuation: if the
same chat already has an active thread, reuse it; otherwise start fresh.
"""

from __future__ import annotations

import json
import logging
import uuid
from typing import TYPE_CHECKING, Any

from redis.asyncio import Redis

if TYPE_CHECKING:
    from aiogram.types import Message

logger = logging.getLogger(__name__)

# Redis keys
_THREAD_KEY = "cupagent:guest_thread:{thread_id}"
_MSG_INDEX_KEY = "cupagent:guest_msg:{chat_id}:{message_id}"
_CHAT_ACTIVE_THREAD_KEY = "cupagent:guest_chat_thread:{chat_id}"

# Limits
_THREAD_TTL = 3 * 24 * 3600  # 3 дня
_MAX_THREAD_MESSAGES = 30


def _new_thread_id() -> str:
    """Generate a new unique thread id."""
    return uuid.uuid4().hex


async def resolve_thread_id(
    redis: Redis,
    message: "Message",
) -> str:
    """Resolve the thread_id for an incoming guest message.

    Order:
        1. If ``reply_to_message`` is set, look up its message_id in the
           index → reuse that thread.
        2. Else if the chat has an active thread → reuse it.
        3. Else → create a new thread_id and mark it active for the chat.

    Args:
        redis: Redis client (decode_responses=True).
        message: Incoming guest Message.

    Returns:
        thread_id string.
    """
    chat_id = message.chat.id if message.chat else 0

    # 1. Reply chain lookup
    reply = message.reply_to_message
    if reply is not None:
        reply_msg_id = reply.message_id
        index_key = _MSG_INDEX_KEY.format(chat_id=chat_id, message_id=reply_msg_id)
        existing = await redis.get(index_key)
        if existing:
            logger.debug(
                "guest_thread: reply chain hit chat=%s msg=%s → thread=%s",
                chat_id, reply_msg_id, existing,
            )
            return existing

    # 2. Chat-scoped active thread
    if chat_id:
        active_key = _CHAT_ACTIVE_THREAD_KEY.format(chat_id=chat_id)
        active = await redis.get(active_key)
        if active:
            logger.debug(
                "guest_thread: active thread for chat=%s → %s", chat_id, active
            )
            return active

    # 3. New thread
    thread_id = _new_thread_id()
    if chat_id:
        active_key = _CHAT_ACTIVE_THREAD_KEY.format(chat_id=chat_id)
        await redis.set(active_key, thread_id, ex=_THREAD_TTL)
    logger.debug("guest_thread: new thread=%s for chat=%s", thread_id, chat_id)
    return thread_id


async def load_thread_history(
    redis: Redis,
    thread_id: str,
) -> list[dict[str, str]]:
    """Load the conversation history for a thread.

    Args:
        redis: Redis client.
        thread_id: Thread identifier.

    Returns:
        List of ``{role, content}`` dicts (empty if none).
    """
    raw = await redis.get(_THREAD_KEY.format(thread_id=thread_id))
    if not raw:
        return []
    try:
        data = json.loads(raw)
        if isinstance(data, list):
            return data
        return []
    except (json.JSONDecodeError, TypeError):
        return []


async def append_to_thread(
    redis: Redis,
    thread_id: str,
    chat_id: int,
    message_id: int,
    role: str,
    content: str,
    *,
    tool_used: bool = False,
) -> None:
    """Append a message to the thread and persist.

    - Adds ``{role, content}`` to the thread's message list.
    - Indexes ``message_id → thread_id`` so future replies link back.
    - Trims to last ``_MAX_THREAD_MESSAGES`` entries.
    - Refreshes TTL on both thread and index keys.

    Args:
        redis: Redis client.
        thread_id: Thread identifier.
        chat_id: Chat id (for index key scoping).
        message_id: Numeric message id to index.
        role: ``"user"`` or ``"assistant"``.
        content: Message text.
        tool_used: If True, marks the assistant turn as having invoked a tool
            (recorded so the LLM knows prices were fetched in-context).
    """
    history = await load_thread_history(redis, thread_id)

    entry: dict[str, Any] = {"role": role, "content": content}
    if role == "assistant" and tool_used:
        entry["tool_used"] = True
    history.append(entry)

    # Trim tail
    history = history[-_MAX_THREAD_MESSAGES:]

    # Save thread
    thread_key = _THREAD_KEY.format(thread_id=thread_id)
    await redis.set(thread_key, json.dumps(history, ensure_ascii=False), ex=_THREAD_TTL)

    # Index this message_id → thread_id
    if chat_id and message_id:
        index_key = _MSG_INDEX_KEY.format(chat_id=chat_id, message_id=message_id)
        await redis.set(index_key, thread_id, ex=_THREAD_TTL)

    # Refresh chat-active thread pointer
    if chat_id:
        active_key = _CHAT_ACTIVE_THREAD_KEY.format(chat_id=chat_id)
        await redis.set(active_key, thread_id, ex=_THREAD_TTL)
