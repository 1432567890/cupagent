"""InitData provider — obtains Telegram WebApp initData via Kurigram.

Uses a user-account MTProto session (Kurigram / Pyrogram fork) to call
`messages.RequestWebView` against a bot's mini-app, which returns a signed
initData string usable for marketplace API auth.

Two cache layers (both optional):
  1. **In-process dict**  — fast path, no I/O, lives with the process.
  2. **Redis**            — survives process restart, shared across workers.

Cache flow on ``get_init_data``:
    in-process dict → Redis → Kurigram RequestWebView

TTL: initData is valid ~24h, so we cache with TTL 23h and refresh on demand
when the remaining lifetime drops below the safety margin.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from typing import Any, Protocol
from urllib.parse import parse_qs

logger = logging.getLogger(__name__)

# Telegram initData validity is ~24h. We cache shorter and refresh early.
_CACHE_TTL = 23 * 3600          # 23h — max time we trust a cached value
_REFRESH_MARGIN = 3600          # refresh if less than 1h remains
_REDIS_KEY = "cupagent:init_data:{bot}"


class InitDataProvider(Protocol):
    """Protocol for anything that can yield Telegram initData."""

    async def get_init_data(self, bot_username: str, app_short_name: str) -> str:
        """Return a valid initData string for the given mini-app bot."""
        ...

    async def close(self) -> None:
        """Release any underlying resources (sessions, clients)."""
        ...


# Known bot user_ids (reference only).  The actual peer resolution for
# RequestAppWebView is done via ``resolve_peer`` at runtime — hardcoded
# access_hashes go stale and cause BOT_INVALID errors.
_KNOWN_BOT_IDS: dict[str, int] = {
    "portals": 7616849367,       # @portals
    "mrkt": 8156315866,          # @mrkt (formerly main_mrkt_bot)
    "main_mrkt_bot": 8156315866, # @main_mrkt_bot — legacy alias of @mrkt
    "grapesmarket_bot": 8395504854,  # @grapesmarket_bot
}

# Default Mini App short_name per bot. The short_name is the part after
# ``?startapp=`` in t.me links and is passed to ``InputBotAppShortName``.
# Override by passing app_short_name explicitly to get_init_data.
_DEFAULT_APP_SHORT_NAMES: dict[str, str] = {
    "portals": "market",
    "mrkt": "app",
    "main_mrkt_bot": "app",
    "grapesmarket_bot": "app",
    "tonnel_network_bot": "tonnel",
    "xgift": "app",
}


def _parse_auth_date(init_data: str) -> int | None:
    """Extract auth_date from an initData query string."""
    if not init_data:
        return None
    params = parse_qs(init_data)
    values = params.get("auth_date", [])
    if not values:
        return None
    try:
        return int(values[0])
    except ValueError:
        return None


def _is_stale(init_data: str) -> bool:
    """True if initData is missing or will expire within the margin."""
    auth_date = _parse_auth_date(init_data)
    if auth_date is None:
        return True
    age = time.time() - auth_date
    return age > 86400 - _REFRESH_MARGIN


class KurigramInitDataProvider:
    """Obtains initData via a Kurigram user session.

    Args:
        api_id, api_hash, session_string: Kurigram/Pyrogram credentials.
        lang_code / system_lang_code: passed to the MTProto client.
        redis: Optional ``redis.asyncio.Redis`` for persistent cross-process
            cache. If ``None``, only the in-process dict is used.
    """

    def __init__(
        self,
        *,
        api_id: int,
        api_hash: str,
        session_string: str | None = None,
        lang_code: str = "ru",
        system_lang_code: str = "ru",
        redis: Any = None,
        session_store: Any = None,
    ) -> None:
        self._api_id = api_id
        self._api_hash = api_hash
        self._session_store = session_store
        # Resolve session string: explicit arg → SessionStore file → env
        self._session_string = self._resolve_session_string(session_string)
        self._lang_code = lang_code
        self._system_lang_code = system_lang_code
        self._redis = redis
        # in-process cache: bot_username -> init_data
        self._cache: dict[str, str] = {}
        self._client: Any = None
        self._lock = asyncio.Lock()

    def _resolve_session_string(self, explicit: str | None) -> str:
        """Resolve the Kurigram session string from multiple sources.

        Order:
            1. Explicit ``session_string`` argument
            2. ``SessionStore`` file at ``user/session/session.string``
            3. ``SESSION_STRING`` environment variable

        Returns:
            The session string, or empty string if none found.
        """
        if explicit:
            return explicit

        if self._session_store is not None:
            stored = self._session_store.load()
            if stored:
                logger.info(
                    "KurigramInitDataProvider: loaded session from %s",
                    self._session_store.path,
                )
                return stored

        env_value = os.environ.get("SESSION_STRING", "")
        if env_value:
            logger.info(
                "KurigramInitDataProvider: loaded session from SESSION_STRING env"
            )
            return env_value

        logger.warning(
            "KurigramInitDataProvider: no session string found "
            "(neither arg, nor file, nor SESSION_STRING env)"
        )
        return ""

    # ── Public API ──────────────────────────────────────────────────────

    async def get_init_data(
        self, bot_username: str, app_short_name: str | None = None
    ) -> str:
        """Return a valid initData for the bot's mini-app.

        Args:
            bot_username: Telegram bot username (without @).
            app_short_name: Mini App short_name (the part after ``?startapp=``
                in t.me links). Falls back to ``_DEFAULT_APP_SHORT_NAMES``.

        Order of resolution:
            1. in-process dict (if not stale)
            2. Redis           (if configured and not stale)
            3. Kurigram RequestAppWebView → write back to both layers
        """
        if app_short_name is None:
            app_short_name = _DEFAULT_APP_SHORT_NAMES.get(
                bot_username.lower(), "app"
            )
        cache_key = f"{bot_username}:{app_short_name}"

        async with self._lock:
            # 1. in-process
            cached = self._cache.get(cache_key)
            if cached and not _is_stale(cached):
                return cached

            # 2. Redis
            if self._redis is not None:
                redis_value = await self._redis.get(
                    _REDIS_KEY.format(bot=cache_key)
                )
                if redis_value and not _is_stale(redis_value):
                    # promote to in-process cache
                    self._cache[cache_key] = redis_value
                    return redis_value

            # 3. Request fresh from Kurigram
            init_data = await self._request_init_data(bot_username, app_short_name)
            self._cache[cache_key] = init_data

            if self._redis is not None:
                try:
                    await self._redis.set(
                        _REDIS_KEY.format(bot=cache_key),
                        init_data,
                        ex=_CACHE_TTL,
                    )
                except Exception:
                    logger.warning(
                        "KurigramInitDataProvider: failed to write Redis cache",
                        exc_info=True,
                    )

            return init_data

    async def close(self) -> None:
        """Stop the Kurigram client. Cache (in-process) is lost on close;
        Redis cache persists for its TTL."""
        if self._client is not None:
            try:
                await self._client.stop()
            except ConnectionError:
                pass
            self._client = None
            logger.info("KurigramInitDataProvider: session stopped")

    async def invalidate_cache(
        self, bot_username: str, app_short_name: str | None = None
    ) -> None:
        """Drop cached initData for a bot so the next call fetches fresh.

        Called by market clients when their auth fails — the cached
        initData may have been rejected server-side despite a recent
        ``auth_date`` (e.g. signature revoked), so we must not reuse it.
        """
        if app_short_name is None:
            app_short_name = _DEFAULT_APP_SHORT_NAMES.get(
                bot_username.lower(), "app"
            )
        cache_key = f"{bot_username}:{app_short_name}"
        async with self._lock:
            self._cache.pop(cache_key, None)
            if self._redis is not None:
                try:
                    await self._redis.delete(
                        _REDIS_KEY.format(bot=cache_key)
                    )
                except Exception:
                    logger.warning(
                        "KurigramInitDataProvider: failed to invalidate "
                        "Redis cache for %s",
                        cache_key,
                        exc_info=True,
                    )

    # ── Kurigram plumbing ──────────────────────────────────────────────

    async def _ensure_client(self) -> Any:
        """Lazily start the Kurigram client using the stored session string.

        After startup, the session may have been refreshed (e.g. by a DC
        migration or key re-generation); we export it and persist back to
        the SessionStore so subsequent restarts reuse the fresh session.
        """
        if self._client is not None:
            return self._client

        if not self._session_string:
            raise RuntimeError(
                "No Kurigram session string available. "
                "Run `python scripts/login.py` to create one, or set SESSION_STRING."
            )

        try:
            from kurigram import Client
        except ImportError:
            try:
                from pyrogram import Client  # type: ignore
            except ImportError as e:
                raise RuntimeError(
                    "Neither kurigram nor pyrogram is installed"
                ) from e

        self._client = Client(
            name="cupagent",
            api_id=self._api_id,
            api_hash=self._api_hash,
            session_string=self._session_string,
            lang_code=self._lang_code,
            system_lang_code=self._system_lang_code,
            in_memory=True,
            no_updates=True,
        )
        await self._client.start()
        logger.info("KurigramInitDataProvider: user session started")

        # Persist potentially-refreshed session back to disk.
        await self._persist_session()
        return self._client

    async def _persist_session(self) -> None:
        """Export the current session string and save it via SessionStore.

        Kurigram/Pyrogram may rotate the auth key on start (DC migration,
        re-auth). Re-exporting keeps the on-disk file in sync so a restart
        doesn't trigger a fresh login.
        """
        if self._session_store is None or self._client is None:
            return

        try:
            fresh = await self._client.export_session_string()
            if fresh and fresh != self._session_string:
                self._session_string = fresh
                self._session_store.save(fresh)
                logger.info(
                    "KurigramInitDataProvider: session refreshed and saved"
                )
        except Exception:
            logger.warning(
                "KurigramInitDataProvider: failed to export/save session",
                exc_info=True,
            )

    async def _request_init_data(
        self, bot_username: str, app_short_name: str
    ) -> str:
        """Open the bot's Mini App via Kurigram and extract initData.

        Uses ``messages.RequestAppWebView`` with an ``InputBotAppShortName``.
        This is the modern Mini App API call that works for bots with
        ``bot_has_main_app=True`` (regardless of ``bot_attach_menu``).

        The returned Web App URL contains a ``tgWebAppData=...`` parameter
        whose value is the signed initData string.

        Args:
            bot_username: Telegram bot username (without @).
            app_short_name: Mini App short_name (after ``?startapp=``).
        """
        from urllib.parse import unquote

        client = await self._ensure_client()

        try:
            from kurigram.raw.functions.messages import RequestAppWebView
            from kurigram.raw.types import InputBotAppShortName
        except ImportError:
            try:
                from pyrogram.raw.functions.messages import RequestAppWebView  # type: ignore
                from pyrogram.raw.types import InputBotAppShortName  # type: ignore
            except ImportError as e:
                raise RuntimeError("kurigram/raw types not available") from e

        # Resolve the bot peer — gives a fresh access_hash every time.
        peer = await client.resolve_peer(bot_username)
        bot_app = InputBotAppShortName(bot_id=peer, short_name=app_short_name)

        web_view = await client.invoke(
            RequestAppWebView(
                peer=peer,
                app=bot_app,
                platform="android",
                write_allowed=True,
            )
        )

        url = getattr(web_view, "url", "") or ""
        if "tgWebAppData=" not in url:
            raise RuntimeError(
                f"RequestAppWebView returned no initData for @{bot_username} "
                f"(short_name={app_short_name!r}): {url[:200]!r}"
            )

        # Extract and URL-decode the initData from the tgWebAppData param.
        init_data = unquote(
            url.split("tgWebAppData=", 1)[1].split("&tgWebAppVersion", 1)[0]
        )
        logger.info(
            "KurigramInitDataProvider: obtained fresh initData for @%s "
            "(app=%r, length=%d)",
            bot_username,
            app_short_name,
            len(init_data),
        )
        return init_data
