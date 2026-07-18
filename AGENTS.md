# AGENTS.md — Guidelines for AI Code Assistants

## Architecture Principles

### Clean & Modular Design
- **Single Responsibility**: Each module does one thing well. No god classes.
- **Dependency Injection**: Pass dependencies via constructor/function args, not globals or singletons.
- **Interfaces & Protocols**: Use `Protocol` (ABC) for abstractions. Markets, caches, databases — all behind interfaces.
- **Separation of Concerns**: Business logic is separate from transport (HTTP, Telegram handlers, DB queries).

### Scalability
- **Async everywhere**: All I/O (HTTP, DB, Redis, Telegram) must be `async`. Never block the event loop.
- **Connection pooling**: Reuse HTTP sessions (`aiohttp.ClientSession`), DB connections (`asyncpg.create_pool`), Redis connections.
- **Graceful shutdown**: Handle SIGINT/SIGTERM — close all connections, flush caches, cancel tasks cleanly.
- **Background tasks**: Use `asyncio.create_task` for periodic jobs (price updates). Track them and cancel on shutdown.

### Code Quality
- **Type hints everywhere**: Every function signature must be fully typed.
- **Docstrings**: Google-style docstrings on all public functions/classes.
- **No magic numbers/strings**: Constants in `const.py` or as class attributes.
- **Error handling**: Define custom exceptions. Handle them at the right layer — don't swallow silently.
- **Logging**: Use `structlog` or standard `logging` with structured output. No bare `print()`.

### Project Structure
```
cupagent/
├── bot/                    # Telegram bot layer (aiogram 3)
│   ├── handlers/           # Message/callback handlers
│   ├── middlewares/         # Auth, logging, error handling
│   └── filters/            # Custom filters
├── markets/                 # Market API clients
│   ├── base.py             # BaseMarketClient protocol + helpers
│   ├── grapesmp/           # Grapes market implementation
│   ├── mrktmp/             # MRKT market implementation
│   └── portalsmp/          # Portal market implementation
├── services/                # Business logic
│   ├── price_service.py     # Price fetching + cache orchestration
│   └── init_data_provider.py # Kurigram initData for market auth
├── user/session/            # Persisted Kurigram string session (.string file)
├── db/                      # Database layer
│   ├── models.py           # SQLAlchemy / asyncpg models
│   ├── repo.py             # Repository pattern
│   └── migrations/          # Alembic migrations
├── cache/                   # Cache layer (Redis)
│   └── redis_cache.py
├── config/                  # Configuration
│   └── settings.py          # pydantic-settings based config
├── core/                    # Shared utilities
│   ├── constants.py
│   ├── exceptions.py
│   └── types.py
├── main.py                  # Entry point
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
└── AGENTS.md                # This file
```

### Testing
- Write tests for business logic and market clients.
- Mock HTTP responses for market API tests.
- Use `pytest-asyncio` for async tests.

### Security
- Never commit `.env` files or tokens.
- Telegram `initData` must be validated server-side (HMAC-SHA256).
- API tokens in config only, never hardcoded.
- Rate-limit outgoing requests to external APIs.

### Git Conventions
- Conventional Commits: `feat:`, `fix:`, `refactor:`, `chore:`, `docs:`.
- Keep PRs small and focused.
