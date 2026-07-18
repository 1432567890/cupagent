"""Start command handler."""

from __future__ import annotations

from aiogram import Router, F
from aiogram.types import (
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    Message,
)

router = Router()

# ── Message templates ─────────────────────────────────────────────────

_START_RU = (
    "пример использования: @cupagentbot что ты умеешь?"
)

_START_EN = (
    "try: @cupagentbot what can you do?"
)

# ── Button labels (localised) ──────────────────────────────────────────

_BUTTON_RU = "попробовать"
_BUTTON_EN = "try it"

# ── Inline query that gets pre-filled in the target chat ────────────────

_INLINE_QUERY_RU = "что ты умеешь?"
_INLINE_QUERY_EN = "what can you do?"


def _is_ru(message: Message) -> bool:
    """Return True if the user's language code is Russian."""
    return (message.from_user.language_code or "").lower().startswith("ru")


def _build_keyboard(message: Message) -> InlineKeyboardMarkup:
    """Build inline keyboard with a localised switch-to-chat query button."""
    if _is_ru(message):
        label = _BUTTON_RU
        query = _INLINE_QUERY_RU
    else:
        label = _BUTTON_EN
        query = _INLINE_QUERY_EN

    return InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(
                    text=label,
                    switch_inline_query=query,
                ),
            ],
        ],
    )


def _pick_template(message: Message) -> str:
    """Choose Russian or English template based on user language code."""
    if _is_ru(message):
        return _START_RU
    return _START_EN


@router.message(F.text.lower() == "/start")
async def cmd_start(message: Message) -> None:
    """Handle /start command — greet the user."""
    text = _pick_template(message)
    keyboard = _build_keyboard(message)
    await message.answer(text, reply_markup=keyboard)
