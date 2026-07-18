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

# GiftWiki API (X-API-Key header)
GIFTWIKI_BASE_URL = "https://api.giftwiki.tg"

# Telegram collectible gift pages (t.me/nft/<slug>-<number>)
TELEGRAM_NFT_BASE_URL = "https://t.me/nft"

# Redis key prefixes
REDIS_KEY_PREFIX = "oclp:floor_price"
REDIS_CRYPTO_KEY = "oclp:crypto:prices"
REDIS_FIAT_KEY_PREFIX = "oclp:fiat"
REDIS_GIFTWIKI_KEY_PREFIX = "oclp:giftwiki"
REDIS_GIFTATTRS_KEY_PREFIX = "oclp:giftattrs"

# Default parse mode for Telegram messages
DEFAULT_PARSE_MODE = "HTML"

# NanoTON multiplier (1 TON = 1_000_000_000 nanoTONs)
NANOTON = 1_000_000_000

# Cache TTLs (seconds)
CRYPTO_PRICES_TTL = 60        # 1 min — crypto is volatile
FIAT_RATES_TTL = 21600        # 6h   — fiat updates once a day
GIFTWIKI_SEARCH_TTL = 600     # 10 min
GIFTWIKI_DETAIL_TTL = 3600    # 1h
GIFTATTRS_TTL = 86400        # 24h — gift attributes are immutable
