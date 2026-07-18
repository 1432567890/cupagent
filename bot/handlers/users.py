"""Admin-only ``/users`` command — reports the total number of tracked users.

Only the two hardcoded admin IDs may run this. For everyone else the
command silently does nothing (no reply), so its existence is not leaked.
"""

from __future__ import annotations

from aiogram import Router, F
from aiogram.types import Message

from db.user_repo import UserRepo

router = Router()

# Telegram user IDs allowed to run /users. Kept inline — admin list is tiny
# and stable. Add more IDs here if needed.
_ADMIN_IDS: frozenset[int] = frozenset({77863476, 36635498})


@router.message(F.text.lower() == "/users")
async def cmd_users(message: Message, user_repo: UserRepo) -> None:
    """Reply with the total number of users in the DB.

    ``user_repo`` is injected from the dispatcher's workflow_data
    (``dp["user_repo"]`` set in main.py). aiogram resolves it by parameter
    name.

    Silently ignores non-admin callers — no reply, no error.
    """
    if message.from_user is None or message.from_user.id not in _ADMIN_IDS:
        return  # not an admin — pretend the command doesn't exist

    try:
        count = await user_repo.count()
    except Exception:
        await message.answer("<i>не удалось получить данные из БД</i>")
        return

    await message.answer(f"👥 Пользователей в базе: <b>{count}</b>")
