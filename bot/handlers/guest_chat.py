"""Guest message handler — answers Telegram Business / managed-chat guests.

Guest messages arrive via ``update.guest_message`` and carry a
``guest_query_id``. The bot is NOT a chat member, so regular ``sendMessage``
does not work. Instead we use ``answerGuestQuery`` to post the reply,
which returns a ``SentGuestMessage`` with an ``inline_message_id``.
That ``inline_message_id`` can then be used with ``editMessageText`` to
implement the "думаю..." → final reply edit UX — exactly like the regular
chat handler but via the guest/inline editing path.

Conversation context is thread-scoped (not per-user): reply chains are
reconstructed via the message_id→thread_id index in
:mod:`bot.handlers._guest_thread`. Each thread holds up to 30 messages
and lives 3 days in Redis.
"""

from __future__ import annotations

import asyncio
import logging
import uuid

from aiogram import Bot, F, Router
from aiogram.enums import ParseMode
from aiogram.types import (
    InlineQueryResultArticle,
    InputTextMessageContent,
    Message,
)
from redis.asyncio import Redis

from bot.handlers._chat_core import (
    FETCHING_PRICES_TEXT,
    LLM_ERROR_TEXT,
    THINKING_TEXT,
    apply_ghost_format_inline,
    build_reply_text,
    extract_reply_context,
    fetching_prices_text,
    is_free_text,
    llm_error_text,
    thinking_text,
)
from bot.handlers._guest_thread import (
    append_to_thread,
    load_thread_history,
    resolve_thread_id,
)
from core.exceptions import LLMError
from services.llm_service import LLMService

logger = logging.getLogger(__name__)

router = Router()


def _build_guest_result(text: str) -> InlineQueryResultArticle:
    """Wrap text into an InlineQueryResultArticle for answerGuestQuery.

    The actual message text is carried by ``input_message_content``.
    parse_mode is NOT set here — we handle formatting in post-processing
    and pass raw HTML (Telegram renders it correctly in guest replies).
    """
    return InlineQueryResultArticle(
        id=str(uuid.uuid4()),
        title="Ответ",
        input_message_content=InputTextMessageContent(
            message_text=text,
            parse_mode=ParseMode.HTML,
            link_preview_options={"is_disabled": True},
        ),
    )


@router.guest_message(F.text.func(is_free_text))
async def handle_guest_chat(
    message: Message,
    bot: Bot,
    llm: LLMService,
    redis: Redis,
    price_service=None,
    crypto_service=None,
    giftwiki_service=None,
    gift_attrs_service=None,
) -> None:
    """Answer a guest free-text message with "думаю..." → edit pattern.

    Uses ``answerGuestQuery`` to post the initial placeholder, then
    ``editMessageText(inline_message_id=...)`` to update to the final reply.
    Falls back to a second ``answerGuestQuery`` if editing fails.

    Bot/llm/redis/price_service/crypto_service/giftwiki_service/
    gift_attrs_service are injected from workflow_data by name.
    """
    guest_query_id = message.guest_query_id
    if guest_query_id is None:
        return

    user_text = message.text or ""
    user_id = message.from_user.id if message.from_user else 0
    chat_id = message.chat.id if message.chat else 0
    message_id = message.message_id
    lang_code = message.from_user.language_code if message.from_user else None

    # Localised status messages.
    _thinking = thinking_text(lang_code)
    _fetching = fetching_prices_text(lang_code)
    _error = llm_error_text(lang_code)

    # Include reply context if the user replied to another message.
    reply_ctx = extract_reply_context(message)
    if reply_ctx:
        user_text = build_reply_text(user_text, reply_ctx)

    # answer_guest_query must be the FIRST call — guest_query_id expires
    # fast (~30-60s, like inline queries), so any Redis work before it
    # risks pushing us past the expiry window and raising
    # "query is too old and response timeout expired".
    try:
        sent = await bot.answer_guest_query(
            guest_query_id=guest_query_id,
            result=_build_guest_result(_thinking),
        )
        inline_message_id = sent.inline_message_id
    except Exception:
        # Query already expired before we could even answer — nothing we
        # can do, the user will need to re-summon the bot. Don't run the
        # LLM (its output would have nowhere to go).
        logger.warning(
            "guest chat: guest_query_id expired before answer for user %d",
            user_id, exc_info=True,
        )
        return

    # Now safe to do Redis-backed context resolution (after the query is
    # answered; subsequent edits use inline_message_id which doesn't expire).
    thread_id = await resolve_thread_id(redis, message)
    history = await load_thread_history(redis, thread_id)

    # Track whether a tool was called and the LLM reply
    tool_used = False
    tool_called = False

    async def _on_tool_call() -> None:
        """Edit placeholder to "fetching data..." on first tool call."""
        nonlocal tool_used, tool_called
        tool_used = True
        if not tool_called:
            tool_called = True
            try:
                await bot.edit_message_text(
                    text=_fetching,
                    inline_message_id=inline_message_id,
                    link_preview_options={"is_disabled": True},
                )
            except Exception:
                pass

    try:
        reply = await llm.chat(
            user_text,
            user_id=user_id,
            history=history,
            price_service=price_service,
            crypto_service=crypto_service,
            giftwiki_service=giftwiki_service,
            gift_attrs_service=gift_attrs_service,
            on_tool_call=_on_tool_call,
        )

        formatted = apply_ghost_format_inline(reply)

        # Edit the guest message to the final reply.
        # NB: we must NOT fall back to a second answerGuestQuery here —
        # guest_query_id expires (~30-60s, like inline queries), so by
        # the time the LLM answers it's already invalid and would raise
        # "query is too old". edit_message_text via inline_message_id has
        # no such expiry, so it's the only viable path for the final reply.
        try:
            await bot.edit_message_text(
                text=formatted,
                inline_message_id=inline_message_id,
                parse_mode=ParseMode.HTML,
                link_preview_options={"is_disabled": True},
            )
        except Exception:
            logger.warning(
                "guest chat: failed to edit guest reply for user %d "
                "(inline_message_id=%s)",
                user_id, inline_message_id, exc_info=True,
            )

        # Persist both turns to the thread
        await append_to_thread(
            redis, thread_id, chat_id, message_id, "user", user_text
        )
        await append_to_thread(
            redis,
            thread_id,
            chat_id,
            message_id,
            "assistant",
            reply,
            tool_used=tool_used,
        )

    except LLMError:
        logger.warning(
            "guest chat: LLM error for user %d thread %s",
            user_id, thread_id, exc_info=True,
        )
        try:
            await bot.edit_message_text(
                text=_error,
                inline_message_id=inline_message_id,
                link_preview_options={"is_disabled": True},
            )
        except Exception:
            pass
    except Exception:
        logger.exception("guest chat handler error")
        try:
            await bot.edit_message_text(
                text=_error,
                inline_message_id=inline_message_id,
                link_preview_options={"is_disabled": True},
            )
        except Exception:
            pass
