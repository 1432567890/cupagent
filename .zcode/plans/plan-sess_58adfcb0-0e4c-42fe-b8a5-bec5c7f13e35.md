# План: команда /users + трекинг юзеров + смягчение рейтинговых лимитов

## A. Смягчение рейтинговых лимитов (1 строка)

**Файл:** `core/constants.py`

Новые значения `RATING_RATE_LIMITS` (вариант «сильно мягче», который вы выбрали):
```python
(0,  0,   60.0),   # level 0  → 1 req / 1 min (было 300)
(1,  1,   20.0),   # level 1  → 1 req / 20 sec (было 60)
(2,  5,    5.0),   # level 2–5 → 1 req / 5 sec (было 10)
(6, 999,   0.0),   # level 6+ → без лимита (было 1)
```

⚠️ Особый случай: cooldown `0.0` означает «без лимита». В `rating_rate_limit.py._check_cooldown` надо явно обработать `cooldown <= 0` → сразу `return True` (не писать в Redis, не блокировать). Сейчас при `0` получилось бы `elapsed < 0` → всегда True, но Redis-запись всё равно делается — это лишняя нагрузка. Добавлю короткий guard.

---

## B. Команда /users + трекинг юзеров

### B1. `db/user_repo.py` (новый файл)

Класс `UserRepo`:
- `__init__(pool: asyncpg.Pool)`
- `init_db()` — `CREATE TABLE IF NOT EXISTS users (user_id BIGINT PRIMARY KEY, first_name TEXT, username TEXT, updated_at TIMESTAMPTZ DEFAULT NOW())`
- `upsert(user_id, first_name, username)` — `INSERT ... ON CONFLICT (user_id) DO UPDATE SET first_name=EXCLUDED.first_name, username=EXCLUDED.username, updated_at=NOW()`
- `count() -> int` — `SELECT COUNT(*) FROM users`

### B2. `bot/middleware/user_tracker.py` (новый файл)

`UserTrackerMiddleware(BaseMiddleware)`:
- `__init__(user_repo)`
- В `__call__`:
  - Не `Message` или `from_user is None` → пропустить.
  - **Fire-and-forget**: `asyncio.create_task(self._safe_upsert(user_id, first_name, username))` — не await, чтобы не блокировать chain. Ошибки логируются внутри таски.

### B3. `bot/handlers/users.py` (новый файл)

```python
_ADMIN_IDS = frozenset({77863476, 36635498})

@router.message(F.text.lower() == "/users")
async def cmd_users(message: Message, user_repo: UserRepo) -> None:
    if message.from_user.id not in _ADMIN_IDS:
        return  # silent for non-admins
    count = await user_repo.count()
    await message.answer(f"👥 Пользователей в базе: <b>{count}</b>")
```

### B4. Подключения

- **`bot/bot.py:create_bot`**: добавить параметр `user_repo: Any = None`. Зарегистрировать `UserTrackerMiddleware(user_repo)` **первым** в `messages_router.message` и `messages_router.guest_message` (перед `WhitelistMiddleware`). `messages_router.include_router(users_router)` рядом с остальными handlers.
- **`main.py:_init_db`**: создать `UserRepo(pool)`, вызвать `await user_repo.init_db()`, вернуть user_repo вместе с floor_price_repo.
- **`main.py:run_bot`**: передать `user_repo=user_repo` в `create_bot(...)` и `dp["user_repo"] = user_repo`.
- **`bot/middleware/__init__.py`**: экспортировать `UserTrackerMiddleware`.

---

## Файлы (итого)
- **Новые:** `db/user_repo.py`, `bot/middleware/user_tracker.py`, `bot/handlers/users.py`
- **Изменённые:** `core/constants.py` (лимиты), `bot/middleware/rating_rate_limit.py` (guard для `cooldown <= 0`), `bot/bot.py`, `main.py`, `bot/middleware/__init__.py`

## Крайние случаи / безопасность
- `/users` для не-админов — silent return (без ответа), чтобы не раскрывать существование команды.
- Upsert в фоновой таске — если БД лежит, бот продолжает работать, ошибка только логируется.
- `username` — nullable (не у всех юзеров есть).
- Middleware первый в chain — ловит вообще всех, кто написал (включая заблокированных antispam), как вы и просили.
- Guard `cooldown <= 0` в rating middleware — пропускает level 6+ без Redis-записей.