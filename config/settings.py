from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables / .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── Telegram Bot ──
    BOT_TOKEN: str = ""

    # ── PostgreSQL ──
    POSTGRES_HOST: str = "localhost"
    POSTGRES_PORT: int = 5432
    POSTGRES_USER: str = "postgres"
    POSTGRES_PASSWORD: str = "postgres"
    POSTGRES_DB: str = "oclp"

    # ── Redis ──
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    REDIS_DB: int = 0
    REDIS_PASSWORD: str = ""

    # ── Market API tokens ──
    # Grapes uses a static API key (X-API-Token header)
    GRAPES_API_TOKEN: str = ""
    # MRKT & Portal use Telegram initData obtained via Kurigram user session
    # (no manual tokens needed — fetched automatically)

    # ── Kurigram user session (for initData) ──
    API_ID: int = 0
    API_HASH: str = ""
    SESSION_STRING: str = ""  # Pyrogram/Kurigram string session
    # Bots whose mini-apps we open to get initData
    GRAPES_BOT_USERNAME: str = "grapesmarket_bot"
    MRKT_BOT_USERNAME: str = "mrkt"
    PORTAL_BOT_USERNAME: str = "portals"  # Portal (@portals) — not to be confused with @portalsmp (channel)
    TONNEL_BOT_USERNAME: str = "Tonnel_Network_bot"
    XGIFT_BOT_USERNAME: str = "xgift"
    # Mini App short_name (the part after ?startapp= in t.me links).
    # Used by messages.RequestAppWebView via InputBotAppShortName.
    MRKT_APP_SHORT_NAME: str = "app"
    PORTAL_APP_SHORT_NAME: str = "market"
    TONNEL_APP_SHORT_NAME: str = "tonnel"
    XGIFT_APP_SHORT_NAME: str = "app"

    # ── GetGems (optional, pre-issued Bearer token from TON Proof) ──
    GETGEMS_API_TOKEN: str = ""

    # ── Price updater ──
    PRICE_UPDATE_INTERVAL: int = 300  # seconds (5 min)

    # ── Whitelist (comma-separated user IDs; empty = open to everyone) ──
    WHITELIST: str = ""

    # ── OpenRouter (LLM) ──
    OPENROUTER_API_KEY: str = ""
    OPENROUTER_MODEL: str = "deepseek/deepseek-v4-flash"
    OPENROUTER_BASE_URL: str = "https://openrouter.ai/api/v1"
    # Max output tokens for the LLM reply.
    OPENROUTER_MAX_TOKENS: int = 1500
    # Comma-separated fallback models tried in order when the primary
    # model returns 429 (rate-limit) or 5xx. Free models (":free" suffix)
    # are rate-limited harder on OpenRouter, so the chain is tried with
    # backoff. Leave empty to disable fallback.
    OPENROUTER_FALLBACK_MODELS: str = (
        "deepseek/deepseek-v4-flash,"
        "google/gemma-4-26b-a4b:free,"
        "google/gemma-4-31b:free,"
        "tencent/hy3:free,"
        "nvidia/nemotron-3-super:free"
    )

    # ── Anti-spam (obvious flood only) ──
    # Only blocks a user who sends the SAME message `threshold` times
    # within `window` seconds. Ordinary conversation (re-asking a
    # question, several different messages) is NEVER blocked — the
    # middleware targets classic spam floods, not legit users.
    ANTISPAM_DUPLICATE_WINDOW: int = 60
    # How many identical messages in the window trigger a hard-block.
    ANTISPAM_DUPLICATE_THRESHOLD: int = 10
    ANTISPAM_BLOCK_SECONDS: int = 120

    # ── GiftWiki API (https://api.giftwiki.tg/docs, X-API-Key header) ──
    GIFTWIKI_TOKEN: str = ""

    # ── Crypto / Fiat APIs (no auth required, override only if proxied) ──
    BINANCE_BASE_URL: str = "https://api.binance.com/api/v3"
    EXCHANGERATE_BASE_URL: str = "https://open.er-api.com/v6"
    CBR_DAILY_URL: str = "https://www.cbr-xml-daily.ru/daily_json.js"

    # ── Derived ──
    @property
    def postgres_dsn(self) -> str:
        return (
            f"postgresql://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}"
            f"@{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"
        )

    @property
    def redis_url(self) -> str:
        password_part = f":{self.REDIS_PASSWORD}@" if self.REDIS_PASSWORD else ""
        return f"redis://{password_part}{self.REDIS_HOST}:{self.REDIS_PORT}/{self.REDIS_DB}"

    @property
    def whitelist_ids(self) -> set[int]:
        """Parse WHITELIST env into a set of ints. Empty = everyone allowed."""
        if not self.WHITELIST.strip():
            return set()
        ids = set()
        for part in self.WHITELIST.split(","):
            part = part.strip()
            if part:
                try:
                    ids.add(int(part))
                except ValueError:
                    continue
        return ids

    @property
    def fallback_models(self) -> list[str]:
        """Parse OPENROUTER_FALLBACK_MODELS into a deduped list.

        Always starts with the primary ``OPENROUTER_MODEL`` (so the chain
        is self-contained), followed by the fallbacks in declared order.
        Empty entries are skipped.
        """
        seen: set[str] = set()
        out: list[str] = []
        candidates = [self.OPENROUTER_MODEL, *self.OPENROUTER_FALLBACK_MODELS.split(",")]
        for m in candidates:
            name = m.strip()
            if name and name not in seen:
                seen.add(name)
                out.append(name)
        return out


@lru_cache
def get_settings() -> Settings:
    return Settings()
