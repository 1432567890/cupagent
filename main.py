#!/usr/bin/env python3
"""Smart entry point — starts the bot, auto-rebuilds Docker on code changes.

When AUTO_REBUILD=true is set, a file watcher triggers `docker compose up --build`
on .py changes. Otherwise, the bot starts directly.

Wires together: PostgreSQL, Redis, Kurigram user session, market clients,
price updater, OpenRouter LLM, and the aiogram bot with whitelist/guest mode.
"""

from __future__ import annotations

import asyncio
import logging
import os
import signal
import sys
from typing import Any

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


async def _init_db(settings) -> tuple[Any, Any, Any]:
    """Initialize database connection pool and create tables.

    Returns ``(floor_price_repo, user_repo, pool)`` so callers can wire
    both repositories into the bot and dispatcher.
    """
    import asyncpg
    from db.repo import FloorPriceRepo
    from db.user_repo import UserRepo

    pool = await asyncpg.create_pool(
        host=settings.POSTGRES_HOST,
        port=settings.POSTGRES_PORT,
        user=settings.POSTGRES_USER,
        password=settings.POSTGRES_PASSWORD,
        database=settings.POSTGRES_DB,
        min_size=2,
        max_size=10,
        # Local Docker postgres runs without SSL — explicitly disable it
        # to avoid TLS-handshake failures (WinError 1225 / ConnectionReset).
        ssl=False,
    )
    floor_price_repo = FloorPriceRepo(pool)
    user_repo = UserRepo(pool)
    await floor_price_repo.init_db()
    await user_repo.init_db()
    return floor_price_repo, user_repo, pool


async def _init_cache(settings) -> tuple[Any, Any]:
    """Initialize Redis connection."""
    import redis.asyncio as aioredis
    from cache.redis_cache import PriceCache

    redis = aioredis.from_url(settings.redis_url, decode_responses=True)
    await redis.ping()
    return PriceCache(redis), redis


async def _init_init_data_provider(settings, redis_client=None) -> Any:
    """Initialize Kurigram user session for initData fetching.

    Required only for MRKT and Portal markets (Grapes uses a static API key).
    Pass the Redis client to persist initData across restarts.

    The Kurigram session string is resolved from (in order):
      1. explicit SESSION_STRING env (settings)
      2. ``user/session/session.string`` file (SessionStore)
    If neither is present, the provider returns None and markets are disabled.
    """
    from user.session.store import SessionStore
    from services.init_data_provider import KurigramInitDataProvider

    store = SessionStore()
    has_env_session = bool(settings.SESSION_STRING)
    has_file_session = store.exists()

    if not has_env_session and not has_file_session:
        logger.warning(
            "No Kurigram session found — MRKT/Portal initData unavailable.\n"
            "  Run `python scripts/login.py` to create one, or set SESSION_STRING."
        )
        return None

    if not settings.API_ID or not settings.API_HASH:
        logger.warning(
            "API_ID/API_HASH not set — cannot start Kurigram client. "
            "Get them at https://my.telegram.org"
        )
        return None

    provider = KurigramInitDataProvider(
        api_id=settings.API_ID,
        api_hash=settings.API_HASH,
        # session_string=None → provider resolves it via SessionStore / env
        session_string=settings.SESSION_STRING or None,
        lang_code="ru",
        system_lang_code="ru",
        redis=redis_client,
        session_store=store,
    )
    return provider


async def _init_llm(settings) -> Any:
    """Initialize OpenRouter LLM client.

    Loads ``instruction.md`` as the base system prompt and the other
    ``skills/*.md`` files into a separate ``skills`` dict. Skills are
    attached to the system prompt lazily per-request (only when the
    user's message matches the skill's trigger keywords), which keeps
    input tokens low on simple queries.
    """
    if not settings.OPENROUTER_API_KEY:
        logger.warning("OPENROUTER_API_KEY not set — LLM chat disabled.")
        return None

    from services.llm_service import LLMService

    skills_dir = "skills"
    system_prompt = ""
    skills: dict[str, str] = {}
    try:
        # Load instruction.md as the always-on base prompt.
        with open(f"{skills_dir}/instruction.md", "r", encoding="utf-8") as f:
            system_prompt = f.read()
        # Load the remaining .md files as lazily-attached skills.
        # Skill name = filename without extension (e.g. "slang" from
        # "slang.md"). These are appended to the system prompt only when
        # the user's message matches the skill's trigger keywords.
        import os

        skill_files = sorted(
            fn for fn in os.listdir(skills_dir)
            if fn.endswith(".md") and fn != "instruction.md"
        )
        for sf in skill_files:
            with open(f"{skills_dir}/{sf}", "r", encoding="utf-8") as f:
                skills[sf[:-3]] = f.read()
    except OSError:
        pass

    logger.info(
        "LLM system prompt: %d chars base + %d skills (%s)",
        len(system_prompt),
        len(skills),
        ", ".join(sorted(skills)) or "none",
    )

    return LLMService(
        api_key=settings.OPENROUTER_API_KEY,
        model=settings.OPENROUTER_MODEL,
        base_url=settings.OPENROUTER_BASE_URL,
        system_prompt=system_prompt,
        max_tokens=settings.OPENROUTER_MAX_TOKENS,
        fallback_models=settings.fallback_models,
        skills=skills or None,
    )


async def _init_crypto_service(settings, redis_client) -> Any:
    """Initialize Crypto & fiat conversion service (no auth needed)."""
    from services.crypto_service import CryptoService

    return CryptoService(
        redis=redis_client,
        binance_base_url=settings.BINANCE_BASE_URL,
        exchangerate_base_url=settings.EXCHANGERATE_BASE_URL,
        cbr_daily_url=settings.CBR_DAILY_URL,
        frankfurter_base_url=settings.FRANKFURTER_BASE_URL,
    )


def _init_giftwiki_service(settings, redis_client) -> Any:
    """Initialize GiftWiki client. Returns None if no token is set."""
    if not settings.GIFTWIKI_TOKEN:
        logger.info(
            "GIFTWIKI_TOKEN not set — GiftWiki collection tools disabled."
        )
        return None

    from services.giftwiki_service import GiftWikiService

    return GiftWikiService(
        api_key=settings.GIFTWIKI_TOKEN,
        redis=redis_client,
    )


def _init_gift_attrs_service(redis_client) -> Any:
    """Initialize the t.me gift-attributes scraper.

    Unlike GiftWiki, this needs no auth token — it reads the public
    ``t.me/nft/<slug>-<number>`` preview page — so it is always on.
    """
    from services.gift_attrs_service import GiftAttrsService

    return GiftAttrsService(redis=redis_client)


def _init_moomin_service(settings, redis_client) -> Any:
    """Initialize Moomin Market API client. Returns None if no key is set."""
    if not settings.MOOMIN_API_KEY:
        logger.info(
            "MOOMIN_API_KEY not set — Moomin snapshot/history tools disabled."
        )
        return None

    from services.moomin_service import MoominService

    return MoominService(
        api_key=settings.MOOMIN_API_KEY,
        redis=redis_client,
    )


async def run_bot() -> None:
    """Initialize all services and start the bot."""
    from config.settings import get_settings
    from bot.bot import create_bot
    from services.price_service import PriceService

    settings = get_settings()

    if not settings.BOT_TOKEN:
        logger.error("BOT_TOKEN is not set. Check your .env file.")
        sys.exit(1)

    # ── DB ──
    logger.info("Initializing database...")
    repo, user_repo, db_pool = await _init_db(settings)

    # ── Redis ──
    logger.info("Initializing Redis cache...")
    cache, redis_client = await _init_cache(settings)

    # ── Kurigram initData provider (for MRKT/Portal) ──
    logger.info("Initializing Kurigram session...")
    init_data_provider = await _init_init_data_provider(settings, redis_client)

    # ── OpenRouter LLM ──
    logger.info("Initializing OpenRouter LLM...")
    llm = await _init_llm(settings)

    # ── Crypto / fiat conversion service ──
    logger.info("Initializing crypto/fiat service...")
    crypto_service = await _init_crypto_service(settings, redis_client)

    # ── GiftWiki (collection lookup, optional) ──
    logger.info("Initializing GiftWiki service...")
    giftwiki_service = _init_giftwiki_service(settings, redis_client)

    # ── Gift attributes scraper (resolves a gift number → model/backdrop) ──
    logger.info("Initializing gift attributes service...")
    gift_attrs_service = _init_gift_attrs_service(redis_client)

    # ── Moomin Market API (cross-market snapshots + OHLC history) ──
    logger.info("Initializing Moomin Market service...")
    moomin_service = _init_moomin_service(settings, redis_client)

    # ── Price service ──
    price_service: PriceService | None = None
    if init_data_provider is not None:
        price_service = PriceService(
            grapes_api_key=settings.GRAPES_API_TOKEN,
            init_data_provider=init_data_provider,
            cache=cache,
            repo=repo,
            update_interval=settings.PRICE_UPDATE_INTERVAL,
            grapes_bot_username=settings.GRAPES_BOT_USERNAME,
            mrkt_bot_username=settings.MRKT_BOT_USERNAME,
            portal_bot_username=settings.PORTAL_BOT_USERNAME,
            tonnel_bot_username=settings.TONNEL_BOT_USERNAME,
            xgift_bot_username=settings.XGIFT_BOT_USERNAME,
            mrkt_app_short_name=settings.MRKT_APP_SHORT_NAME,
            portal_app_short_name=settings.PORTAL_APP_SHORT_NAME,
            tonnel_app_short_name=settings.TONNEL_APP_SHORT_NAME,
            xgift_app_short_name=settings.XGIFT_APP_SHORT_NAME,
            getgems_api_token=settings.GETGEMS_API_TOKEN,
        )
        await price_service.start_periodic_updates()
    else:
        logger.warning(
            "PriceService disabled — no Kurigram session for initData."
        )

    # ── Bot ──
    whitelist = settings.whitelist_ids or None
    bot, dp = create_bot(
        settings.BOT_TOKEN,
        whitelist=whitelist,
        redis=redis_client,
        user_repo=user_repo,
        antispam_duplicate_window=settings.ANTISPAM_DUPLICATE_WINDOW,
        antispam_duplicate_threshold=settings.ANTISPAM_DUPLICATE_THRESHOLD,
        antispam_block_seconds=settings.ANTISPAM_BLOCK_SECONDS,
    )

    # Inject services into handler context
    dp["price_service"] = price_service
    dp["llm"] = llm
    dp["redis"] = redis_client
    dp["user_repo"] = user_repo
    dp["crypto_service"] = crypto_service
    dp["giftwiki_service"] = giftwiki_service
    dp["gift_attrs_service"] = gift_attrs_service
    dp["moomin_service"] = moomin_service

    # ── Graceful shutdown ──
    async def _shutdown(signum: int) -> None:
        logger.info("Shutting down (signal=%d)...", signum)
        if price_service:
            await price_service.close()
        if llm:
            await llm.close()
        if crypto_service:
            await crypto_service.close()
        if giftwiki_service:
            await giftwiki_service.close()
        if gift_attrs_service:
            await gift_attrs_service.close()
        if moomin_service:
            await moomin_service.close()
        if init_data_provider is not None:
            await init_data_provider.close()
        await db_pool.close()
        await redis_client.close()
        await bot.session.close()
        await dp.stop_polling()
        sys.exit(0)

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(
            sig, lambda s=sig: asyncio.ensure_future(_shutdown(s))
        )

    logger.info(
        "Starting bot polling (whitelist=%s, llm=%s, prices=%s, "
        "crypto=%s, giftwiki=%s, gift_attrs=%s, moomin=%s)...",
        "open" if whitelist is None else f"{len(whitelist)} users",
        "on" if llm else "off",
        "on" if price_service else "off",
        "on" if crypto_service else "off",
        "on" if giftwiki_service else "off",
        "on" if gift_attrs_service else "off",
        "on" if moomin_service else "off",
    )
    try:
        await dp.start_polling(
            bot, allowed_updates=dp.resolve_used_update_types()
        )
    finally:
        if price_service:
            await price_service.close()
        if llm:
            await llm.close()
        if crypto_service:
            await crypto_service.close()
        if giftwiki_service:
            await giftwiki_service.close()
        if gift_attrs_service:
            await gift_attrs_service.close()
        if moomin_service:
            await moomin_service.close()
        if init_data_provider is not None:
            await init_data_provider.close()
        await db_pool.close()
        await redis_client.close()
        await bot.session.close()


def main() -> None:
    """Entry point."""
    if os.getenv("AUTO_REBUILD", "false").lower() == "true":
        _run_with_rebuild()
    else:
        asyncio.run(run_bot())


def _run_with_rebuild() -> None:
    """Run with a file watcher that triggers Docker rebuild on .py changes."""
    import subprocess

    try:
        from watchdog.observers import Observer
        from watchdog.events import FileSystemEventHandler
    except ImportError:
        logger.warning(
            "watchdog not installed. Install with: pip install watchdog"
        )
        logger.info("Falling back to direct bot startup.")
        asyncio.run(run_bot())
        return

    class RestartHandler(FileSystemEventHandler):
        """Triggers Docker rebuild on Python file changes."""

        def __init__(self) -> None:
            super().__init__()
            self._loop = asyncio.new_event_loop()
            self._debounce_task: asyncio.Task | None = None

        def on_modified(self, event) -> None:
            if event.src_path.endswith(".py") and "__pycache__" not in event.src_path:
                logger.info("Detected change: %s", event.src_path)
                if self._debounce_task:
                    self._debounce_task.cancel()
                self._debounce_task = self._loop.create_task(
                    self._rebuild_after_delay()
                )

        async def _rebuild_after_delay(self) -> None:
            """Wait 2s to debounce rapid saves, then rebuild."""
            await asyncio.sleep(2)
            logger.info("Triggering Docker rebuild...")
            try:
                subprocess.run(
                    ["docker", "compose", "up", "--build", "-d"],
                    check=True,
                    capture_output=True,
                    text=True,
                )
                logger.info("Docker rebuild complete.")
            except subprocess.CalledProcessError as e:
                logger.error("Docker rebuild failed: %s", e.stderr)

    observer = Observer()
    handler = RestartHandler()
    observer.schedule(handler, path=".", recursive=True)
    observer.start()
    logger.info("File watcher active — will rebuild Docker on .py changes")

    try:
        asyncio.run(run_bot())
    finally:
        observer.stop()
        observer.join()


if __name__ == "__main__":
    main()
