from __future__ import annotations

# Market API base URLs
GRAPES_BASE_URL = "https://api.grapesmarket.xyz"
MRKT_BASE_URL = "https://api.tgmrkt.io"
PORTAL_BASE_URL = "https://portal-market.com"
GETGEMS_BASE_URL = "https://api.getgems.io/public-api"
TONNEL_BASE_URL = "https://gifts2.tonnel.network/api"
XGIFT_BASE_URL = "https://app-api.xgift.tg"

# Crypto / Fiat APIs (no auth required)
BINANCE_BASE_URL = "https://api.binance.com/api/v3"
EXCHANGERATE_BASE_URL = "https://open.er-api.com/v6"
CBR_DAILY_URL = "https://www.cbr-xml-daily.ru/daily_json.js"
# Frankfurter (ECB open rates, no auth) — used for fiat history.
# NOTE: ECB dropped RUB in 2022, so RUB/UAH/KZT/GEL/BYN are NOT here —
# those fall back to the CBR archive (cbr-xml-daily.ru/archive/...).
FRANKFURTER_BASE_URL = "https://api.frankfurter.app"
# CBR historical archive — one JSON file per business day.
# Template: {CBR_ARCHIVE_URL_TEMPLATE.format(year=..., month=..., day=...)}
CBR_ARCHIVE_URL_TEMPLATE = (
    "https://www.cbr-xml-daily.ru/archive/"
    "{year:04d}/{month:02d}/{day:02d}/daily_json.js"
)

# GiftWiki API (X-API-Key header)
GIFTWIKI_BASE_URL = "https://api.giftwiki.tg"

# Moomin Market API (X-API-Key header) — cross-market gift collection aggregator.
# Quote asset is TON (= GRAM 1:1). Slugs are lowercase alphanumeric
# versions of the collection prefix: "Artisan Brick" -> "artisanbrick".
MOOMIN_BASE_URL = "https://api.moomin.cfd/market/v1"

# Telegram collectible gift pages (t.me/nft/<slug>-<number>)
TELEGRAM_NFT_BASE_URL = "https://t.me/nft"

# Redis key prefixes
REDIS_KEY_PREFIX = "cupagent:floor_price"
REDIS_CRYPTO_KEY = "cupagent:crypto:prices"
REDIS_FIAT_KEY_PREFIX = "cupagent:fiat"
# Crypto kline history — historical OHLC bars (Binance). Cached per
# (symbol, interval, days). Bars are append-only historical data, so a
# short TTL is fine.
REDIS_CRYPTO_HISTORY_KEY_PREFIX = "cupagent:crypto:history"
# Fiat history — daily rate series (Frankfurter / CBR archive).
REDIS_FIAT_HISTORY_KEY_PREFIX = "cupagent:fiat:history"
REDIS_GIFTWIKI_KEY_PREFIX = "cupagent:giftwiki"
REDIS_GIFTATTRS_KEY_PREFIX = "cupagent:giftattrs"
REDIS_MOOMIN_KEY_PREFIX = "cupagent:moomin"

# Default parse mode for Telegram messages
DEFAULT_PARSE_MODE = "HTML"

# NanoTON multiplier (1 TON = 1_000_000_000 nanoTONs)
NANOTON = 1_000_000_000

# Cache TTLs (seconds)
CRYPTO_PRICES_TTL = 60        # 1 min — crypto is volatile
FIAT_RATES_TTL = 21600        # 6h   — fiat updates once a day
GIFTWIKI_DETAIL_TTL = 3600    # 1h
GIFTATTRS_TTL = 86400        # 24h — gift attributes are immutable

# Currency-history TTLs. Crypto klines are historical (append-only), so a
# few minutes is plenty. Fiat daily series are immutable past dates, so
# they can be cached for a day.
CRYPTO_HISTORY_TTL = 300      # 5m — OHLC bars
FIAT_HISTORY_TTL = 86400      # 24h — past daily rates are immutable

# Moomin Market API cache TTLs.
# Collections list changes rarely (new drops only) — cache aggressively.
# Snapshot prices are volatile — short TTL. Candle bars are historical
# and append-only, so a few minutes is fine.
MOOMIN_COLLECTIONS_TTL = 3600   # 1h
MOOMIN_SNAPSHOT_TTL = 60        # 1m  — prices are volatile
MOOMIN_CANDLES_TTL = 300        # 5m  — historical bars
MOOMIN_HTTP_TIMEOUT = 20        # seconds

# Rating-based rate-limit tiers (Telegram Star spending level → cooldown).
# Each entry maps a rating level range to the minimum seconds between
# consecutive free-text (LLM) requests from that user.
# Accessed via ``ChatFullInfo.rating.level`` (int, 0+).
# Lower level → stricter limit; higher level → more generous.
RATING_RATE_LIMITS: list[tuple[int, int, float]] = [
    # (min_level, max_level, cooldown_seconds)
    (0,  0,   60.0),   # level 0  → 1 request per 1 min
    (1,  1,   20.0),   # level 1  → 1 request per 20 sec
    (2,  5,    5.0),   # level 2–5 → 1 request per 5 sec
    (6, 999,   0.0),   # level 6+  → no limit (0 = disabled)
]

# Redis key for storing the timestamp of a user's last allowed request.
# {user_id} is replaced with the numeric Telegram user ID.
RATING_RATE_LIMIT_KEY = "cupagent:rating_rate:{user_id}"

# Redis key for caching a user's rating level to avoid calling get_chat
# on every single message.
RATING_LEVEL_CACHE_KEY = "cupagent:rating_level:{user_id}"

# TTL for the rating-level cache (seconds).
RATING_LEVEL_CACHE_TTL = 3600  # 1 hour
