"""LLM chat handler for regular messages — passes free-text to OpenRouter.

Only fires for whitelisted users (middleware sets ``whitelisted=True``).
Shows a "думаю…" status message that is edited to the final reply once
the model answers. Supports tool-calling via :class:`PriceService` with
live UI feedback.

Guest messages (Telegram Business / managed chats) are handled separately
in :mod:`bot.handlers.guest_chat`.
"""

from __future__ import annotations

import asyncio
import logging

from aiogram import Bot, F, Router
from aiogram.types import Message
from redis.asyncio import Redis

from bot.handlers._chat_core import (
    FETCHING_PRICES_TEXT,
    LLM_ERROR_TEXT,
    THINKING_TEXT,
    apply_ghost_format,
    generate_reply,
    is_free_text,
    typing_action_loop,
)
from core.exceptions import LLMError
from services.llm_service import LLMService
from services.price_service import PriceService

logger = logging.getLogger(__name__)

router = Router()


@router.message(F.text.func(is_free_text))
async def handle_chat(
    message: Message,
    bot: Bot,
    llm: LLMService,
    redis: Redis,
    price_service: PriceService | None = None,
    crypto_service=None,
    giftwiki_service=None,
    gift_attrs_service=None,
) -> None:
    """Forward free-text messages to the LLM with typing preview.

    ``bot``, ``llm``, ``redis``, ``price_service``, ``crypto_service``,
    ``giftwiki_service`` and ``gift_attrs_service`` are
    injected from the dispatcher's workflow_data (set in main.py via
    ``dp["key"] = value``).
    """
    user_text = message.text or ""
    user_id = message.from_user.id if message.from_user else 0
    chat_id = message.chat.id

    # Send initial "думаю..." placeholder as a reply to the user message
    status_msg = await message.answer(
        THINKING_TEXT,
        reply_to_message_id=message.message_id,
        link_preview_options={"is_disabled": True},
    )

    # Keep typing indicator alive while we wait
    typing_task = asyncio.create_task(
        typing_action_loop(bot, chat_id)
    )

    # Flag to track if we already showed "получаю данные" status
    tool_called = False

    async def _on_tool_call() -> None:
        """Callback invoked by LLMService when a tool is about to be called."""
        nonlocal tool_called
        if not tool_called:
            tool_called = True
            try:
                await status_msg.edit_text(FETCHING_PRICES_TEXT)
            except Exception:
                pass

    try:
        reply = await generate_reply(
            llm,
            redis,
            user_id,
            user_text,
            price_service=price_service,
            crypto_service=crypto_service,
            giftwiki_service=giftwiki_service,
            gift_attrs_service=gift_attrs_service,
            on_tool_call=_on_tool_call,
        )

        # Edit the status message to the final reply
        try:
            await status_msg.edit_text(
                apply_ghost_format(reply),
                link_preview_options={"is_disabled": True},
            )
        except Exception:
            # Fallback: delete status and send new message
            try:
                await status_msg.delete()
            except Exception:
                pass
            await message.answer(
                apply_ghost_format(reply),
                link_preview_options={"is_disabled": True},
            )

    except LLMError:
        logger.warning("chat handler: LLM error for user %d", user_id, exc_info=True)
        try:
            await status_msg.edit_text(
                LLM_ERROR_TEXT,
                link_preview_options={"is_disabled": True},
            )
        except Exception:
            pass
    except Exception:
        logger.exception("chat handler error")
        try:
            await status_msg.edit_text(
                LLM_ERROR_TEXT,
                link_preview_options={"is_disabled": True},
            )
        except Exception:
            pass
    finally:
        typing_task.cancel()
