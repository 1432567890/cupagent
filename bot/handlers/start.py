"""Start command handler."""

from __future__ import annotations

from aiogram import Router, F
from aiogram.types import Message

router = Router()


@router.message(F.text.lower() == "/start")
async def cmd_start(message: Message) -> None:
    """Handle /start command — greet the user."""
    await message.answer(
        "Привет! 👋\n\n"
        "Я <b>oclp</b> — бот для мониторинга цен подарков на маркетплейсах.\n\n"
        "Доступные команды:\n"
        "• /start — приветствие\n"
        "• /prices — текущие флор-цены\n"
        "• /prices grapes — цены на Grapes\n"
        "• /prices mrkt — цены на MRKT\n"
        "• /prices portal — цены на Portal"
    )
