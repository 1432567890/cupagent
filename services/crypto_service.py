"""Crypto & fiat currency conversion service.

Fetches live crypto prices from Binance public API (no auth) and fiat
exchange rates from ExchangeRate-API (open.er-api.com), with a fallback
to the Central Bank of Russia JSON wrapper (cbr-xml-daily.ru).

All responses are cached in Redis to minimize external requests and
LLM input tokens (the LLM tool returns compact JSON).

Architecture (per AGENTS.md):
    - CryptoPriceProvider Protocol for abstraction / future fallbacks.
    - CryptoService with constructor DI (keyword-only args).
    - aiohttp.ClientSession reused lazily via property.
    - All I/O is async; never blocks the event loop.
"""

from __future__ import annotations

import asyncio
import json
import logging
from decimal import Decimal, InvalidOperation
from typing import TYPE_CHECKING, Any, Protocol

import aiohttp

from core.constants import (
    CBR_DAILY_URL,
    CRYPTO_PRICES_TTL,
    EXCHANGERATE_BASE_URL,
    FIAT_RATES_TTL,
    REDIS_CRYPTO_KEY,
    REDIS_FIAT_KEY_PREFIX,
)
from core.exceptions import CupagentError

if TYPE_CHECKING:
    from redis.asyncio import Redis

logger = logging.getLogger(__name__)


class CryptoError(CupagentError):
    """Crypto / fiat conversion error."""


def _round(value: Decimal, places: int) -> Decimal:
    """Round a Decimal to ``places`` decimal places (no exponent)."""
    if not isinstance(value, Decimal) or value.is_nan():
        return Decimal("0")
    quant = Decimal(1).scaleb(-places)  # 10**-places
    return value.quantize(quant)


class CryptoPriceProvider(Protocol):
    """Abstract crypto price source (for tests / future providers)."""

    async def get_crypto_prices(self, symbols: list[str]) -> dict[str, Decimal]:
        ...

    async def get_fiat_rates(self, base: str = "USD") -> dict[str, Decimal]:
        ...


# ── Alias maps ────────────────────────────────────────────────────────────
# Accept Russian/English slang and lowercase variants from the LLM,
# normalize to canonical Binance symbol or fiat ISO code.

_CRYPTO_ALIASES: dict[str, str] = {
    # Bitcoin
    "btc": "BTC", "bitcoin": "BTC", "биткоин": "BTC", "биток": "BTC",
    # Ethereum
    "eth": "ETH", "ethereum": "ETH", "эфир": "ETH", "эфириум": "ETH",
    # TON (Telegram Open Network)
    "ton": "TON", "toncoin": "TON", "тон": "TON", "тонкоин": "TON",
    # GRAM (post-rebrand of TON, 1:1)
    "gram": "GRAM", "grams": "GRAM", "грам": "GRAM",
    # USDT
    "usdt": "USDT", "tether": "USDT", "тезер": "USDT", "тетер": "USDT",
    # Solana
    "sol": "SOL", "solana": "SOL",
    # Tron
    "trx": "TRX", "tron": "TRX",
    # Notcoin
    "not": "NOT", "notcoin": "NOT", "ноткоин": "NOT",
    # Dogs
    "dogs": "DOGS", "догс": "DOGS",
}

_FIAT_ALIASES: dict[str, str] = {
    "usd": "USD", "доллар": "USD", "долл": "USD", "бакс": "USD",
    "rub": "RUB", "rur": "RUB", "руб": "RUB", "рубль": "RUB", "рубли": "RUB",
    "eur": "EUR", "euro": "EUR", "евро": "EUR",
    "gbp": "GBP", "фунт": "GBP",
    "cny": "CNY", "юань": "CNY",
    "uah": "UAH", "гривна": "UAH",
    "kzt": "KZT", "тенге": "KZT",
    "gel": "GEL", "лари": "GEL",
    "byn": "BYN", "белруб": "BYN",
    "try": "TRY", "лира": "TRY",
    "aed": "AED", "дирхам": "AED",
    "inr": "INR", "рупия": "INR",
}

# Fiat currencies we know about (everything else is treated as crypto)
_KNOWN_FIAT: set[str] = {
    "USD", "EUR", "GBP", "RUB", "CNY", "UAH", "KZT", "GEL", "BYN",
    "TRY", "AED", "INR", "JPY", "KRW", "CHF", "CAD", "AUD", "PLN",
    "CZK", "SEK", "NOK", "DKK", "HKD", "SGD", "THB", "MYR", "PHP",
    "BRL", "MXN", "ZAR", "UZS", "KGS",
}


def normalize_asset(raw: str) -> tuple[str, str]:
    """Resolve user-facing asset string to ``(canonical, kind)``.

    Args:
        raw: User/LLM input, e.g. ``"грам"``, ``"BTC"``, ``"руб"``.

    Returns:
        Tuple of (canonical symbol, "crypto"|"fiat"|"unknown").

    Examples:
        >>> normalize_asset("грам")
        ('GRAM', 'crypto')
        >>> normalize_asset("руб")
        ('RUB', 'fiat')
    """
    key = raw.strip().lower()
    if key in _FIAT_ALIASES:
        return _FIAT_ALIASES[key], "fiat"
    if key in _CRYPTO_ALIASES:
        return _CRYPTO_ALIASES[key], "crypto"
    # Uppercase 3-letter codes
    up = raw.strip().upper()
    if up in _KNOWN_FIAT:
        return up, "fiat"
    if len(up) >= 2 and up.isalpha():
        return up, "crypto"
    return raw.strip(), "unknown"


class CryptoService:
    """Live crypto + fiat rates with Redis caching.

    Crypto prices come from Binance (pair ``<ASSET>USDT``). Fiat rates
    come from ExchangeRate-API with a fallback to the Central Bank of
    Russia. All responses are cached to minimize external calls and to
    keep LLM tool results compact.

    Note on GRAM: post-rebrand, GRAM may not yet be listed on Binance.
    In that case we fall back to TON (1:1) and log a warning.
    """

    # Default symbols we pre-fetch on startup (for proactive cache warm).
    _DEFAULT_CRYPTO_SYMBOLS: tuple[str, ...] = (
        "BTCUSDT", "ETHUSDT", "TONUSDT", "SOLUSDT", "TRXUSDT",
    )

    def __init__(
        self,
        *,
        redis: Redis,
        binance_base_url: str = "https://api.binance.com/api/v3",
        exchangerate_base_url: str = "https://open.er-api.com/v6",
        cbr_daily_url: str = "https://www.cbr-xml-daily.ru/daily_json.js",
    ) -> None:
        self._redis = redis
        self._binance_base = binance_base_url.rstrip("/")
        self._fiat_base = exchangerate_base_url.rstrip("/")
        self._cbr_url = cbr_daily_url
        self._session: aiohttp.ClientSession | None = None

    @property
    def session(self) -> aiohttp.ClientSession:
        """Lazily-initialized shared HTTP session."""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=15),
                headers={"Accept": "application/json"},
            )
        return self._session

    # ── Crypto (Binance) ──────────────────────────────────────────────

    async def get_crypto_prices(
        self, symbols: list[str]
    ) -> dict[str, Decimal]:
        """Fetch crypto prices in USDT for the given Binance symbols.

        Args:
            symbols: Binance pair symbols, e.g. ``["BTCUSDT", "TONUSDT"]``.

        Returns:
            Mapping ``{"BTCUSDT": Decimal("63929.81"), ...}``. Missing
            or unavailable symbols are silently dropped.
        """
        symbols = [s for s in symbols if s]
        if not symbols:
            return {}

        # Try cache first (hash: symbol -> price string)
        cached = await self._redis.hgetall(REDIS_CRYPTO_KEY)
        missing = [s for s in symbols if s not in cached]
        out: dict[str, Decimal] = {}
        for s in symbols:
            if s in cached:
                try:
                    out[s] = Decimal(cached[s])
                except InvalidOperation:
                    pass
        if not missing:
            return out

        # Fetch missing from Binance batch endpoint
        fetched = await self._fetch_binance_prices(missing)
        out.update(fetched)

        if fetched:
            # Cache all fetched prices (string values) with TTL
            pipe = self._redis.pipeline()
            pipe.hset(
                REDIS_CRYPTO_KEY,
                mapping={k: str(v) for k, v in fetched.items()},
            )
            pipe.expire(REDIS_CRYPTO_KEY, CRYPTO_PRICES_TTL)
            await pipe.execute()
        return out

    async def _fetch_binance_prices(
        self, symbols: list[str]
    ) -> dict[str, Decimal]:
        """Call Binance /ticker/price batch endpoint.

        Uses URL-encoded JSON array of symbols. Returns a map of
        ``symbol -> Decimal(price)``. Failed / unknown symbols are
        dropped silently (logged at debug level).
        """
        # GRAM fallback: if GRAMUSDT is requested but not available,
        # substitute TONUSDT (GRAM is the rebranded TON, 1:1 ratio).
        gram_requested = "GRAMUSDT" in symbols
        fetch_symbols = list(symbols)
        if gram_requested and "TONUSDT" not in fetch_symbols:
            fetch_symbols.append("TONUSDT")

        # Validate symbols are safe (ASCII alphanumeric only) to prevent
        # URL injection — Binance symbols look like BTCUSDT, ETHUSDT.
        safe_symbols = [
            s for s in fetch_symbols
            if isinstance(s, str) and s.isascii() and s.isalnum()
        ]
        if not safe_symbols:
            return {}
        fetch_symbols = safe_symbols

        # Binance batch endpoint expects a compact JSON array with NO
        # whitespace: ["BTCUSDT","ETHUSDT"]. aiohttp would URL-encode
        # the brackets/quotes which Binance rejects, so we build the URL
        # ourselves with literal square brackets (allowed unencoded).
        symbols_json = json.dumps(fetch_symbols, separators=(",", ":"))
        url = f"{self._binance_base}/ticker/price?symbols={symbols_json}"
        try:
            async with self.session.get(url) as resp:
                if resp.status != 200:
                    body = await resp.text()
                    logger.warning(
                        "CryptoService: Binance HTTP %d: %s",
                        resp.status, body[:200],
                    )
                    return {}
                data = await resp.json()
        except aiohttp.ClientError as e:
            logger.warning("CryptoService: Binance network error: %s", e)
            return {}

        out: dict[str, Decimal] = {}
        for item in data:
            sym = item.get("symbol", "")
            try:
                out[sym] = Decimal(str(item.get("price", "0")))
            except (InvalidOperation, TypeError):
                continue

        # Map TON -> GRAM if GRAM was requested but unavailable
        if gram_requested and "GRAMUSDT" not in out and "TONUSDT" in out:
            out["GRAMUSDT"] = out["TONUSDT"]
            logger.info("CryptoService: GRAMUSDT not listed, using TONUSDT")
        # Drop the TONUSDT entry if user didn't ask for it
        if gram_requested and "TONUSDT" not in symbols:
            out.pop("TONUSDT", None)
        return out

    # ── Fiat (ExchangeRate-API, fallback CBR) ─────────────────────────

    async def get_fiat_rates(self, base: str = "USD") -> dict[str, Decimal]:
        """Fetch fiat exchange rates for ``base`` currency.

        Args:
            base: ISO 4217 base code (default USD).

        Returns:
            Mapping of currency code -> rate. Includes RUB.
        """
        redis_key = f"{REDIS_FIAT_KEY_PREFIX}:{base}"
        cached = await self._redis.get(redis_key)
        if cached:
            try:
                return {k: Decimal(v) for k, v in json.loads(cached).items()}
            except (json.JSONDecodeError, InvalidOperation):
                pass

        rates = await self._fetch_exchangerate(base)
        if not rates:
            rates = await self._fetch_cbr_fallback()

        if rates:
            await self._redis.set(
                redis_key,
                json.dumps({k: str(v) for k, v in rates.items()}),
                ex=FIAT_RATES_TTL,
            )
        return rates

    async def _fetch_exchangerate(self, base: str) -> dict[str, Decimal]:
        """Fetch rates from open.er-api.com (no auth, includes RUB)."""
        url = f"{self._fiat_base}/latest/{base}"
        try:
            async with self.session.get(url) as resp:
                if resp.status != 200:
                    logger.warning(
                        "CryptoService: ExchangeRate HTTP %d", resp.status
                    )
                    return {}
                data = await resp.json()
        except aiohttp.ClientError as e:
            logger.warning("CryptoService: ExchangeRate error: %s", e)
            return {}

        if data.get("result") != "success":
            logger.warning("CryptoService: ExchangeRate bad payload")
            return {}

        rates_raw = data.get("rates", {})
        out: dict[str, Decimal] = {}
        for code, val in rates_raw.items():
            try:
                out[code.upper()] = Decimal(str(val))
            except (InvalidOperation, TypeError):
                continue
        return out

    async def _fetch_cbr_fallback(self) -> dict[str, Decimal]:
        """Fallback to Central Bank of Russia (rates are RUB-based)."""
        try:
            async with self.session.get(self._cbr_url) as resp:
                if resp.status != 200:
                    return {}
                data = await resp.json(content_type=None)
        except aiohttp.ClientError as e:
            logger.warning("CryptoService: CBR error: %s", e)
            return {}

        # CBR gives per-unit-in-RUB. Convert to USD base.
        valute = data.get("Valute", {})
        rub_per_usd = valute.get("USD", {}).get("Value")
        if not rub_per_usd:
            return {}
        rub_per_usd = Decimal(str(rub_per_usd))

        # USD base = 1, then for each currency: rate = rub_per_unit / rub_per_usd
        out: dict[str, Decimal] = {"USD": Decimal("1"), "RUB": rub_per_usd}
        for code, info in valute.items():
            try:
                nominal = Decimal(str(info.get("Nominal", 1)))
                value = Decimal(str(info["Value"]))
                # value is for `nominal` units of the currency
                per_unit_rub = value / nominal
                out[code] = per_unit_rub / rub_per_usd
            except (InvalidOperation, KeyError, TypeError):
                continue
        return out

    # ── High-level convert ────────────────────────────────────────────

    async def convert(
        self, amount: Decimal | float | int,
        from_asset: str, to_asset: str,
    ) -> dict[str, Any]:
        """Convert ``amount`` from one asset to another.

        Handles crypto→crypto, crypto→fiat, fiat→fiat, fiat→crypto.

        Args:
            amount: Quantity to convert.
            from_asset: Source asset (alias or symbol).
            to_asset: Target asset (alias or symbol).

        Returns:
            Compact dict: ``{"amount": float, "from": "GRAM",
            "to": "RUB", "result": float, "rate": float}``.
            On error returns ``{"error": "..."}``.
        """
        try:
            amt = Decimal(str(amount))
        except (InvalidOperation, TypeError):
            return {"error": f"invalid amount: {amount}"}

        from_canon, from_kind = normalize_asset(from_asset)
        to_canon, to_kind = normalize_asset(to_asset)

        if from_kind == "unknown" or to_kind == "unknown":
            return {"error": f"unknown asset: {from_asset!r} or {to_asset!r}"}
        if amt < 0:
            return {"error": "amount must be >= 0"}

        # Same currency shortcut
        if from_canon == to_canon:
            return self._result(amt, from_canon, to_canon, Decimal("1"))

        # Fiat -> Fiat
        if from_kind == "fiat" and to_kind == "fiat":
            rates = await self.get_fiat_rates("USD")
            return self._convert_via_usd(
                amt, from_canon, to_canon, rates
            )

        # Crypto -> Crypto
        if from_kind == "crypto" and to_kind == "crypto":
            prices = await self.get_crypto_prices(
                [f"{from_canon}USDT", f"{to_canon}USDT"]
            )
            p_from = prices.get(f"{from_canon}USDT")
            p_to = prices.get(f"{to_canon}USDT")
            if not p_from or not p_to:
                return {"error": f"no price for {from_canon} or {to_canon}"}
            rate = p_from / p_to
            return self._result(amt, from_canon, to_canon, rate)

        # Crypto -> Fiat or Fiat -> Crypto: route via USD
        rates = await self.get_fiat_rates("USD")
        crypto_sym = from_canon if from_kind == "crypto" else to_canon
        prices = await self.get_crypto_prices([f"{crypto_sym}USDT"])
        p_crypto = prices.get(f"{crypto_sym}USDT")
        if not p_crypto:
            return {"error": f"no Binance price for {crypto_sym}USDT"}

        if from_kind == "crypto":
            # crypto -> USD (price) -> fiat
            usd_amount = amt * p_crypto
            return self._convert_via_usd(usd_amount, "USD", to_canon, rates)
        else:
            # fiat -> USD -> crypto
            usd_amount = self._to_usd(amt, from_canon, rates)
            if usd_amount is None:
                return {"error": f"no fiat rate for {from_canon}"}
            result = usd_amount / p_crypto
            rate = result / amt if amt > 0 else Decimal("0")
            return self._result(amt, from_canon, to_canon, rate)

    def _convert_via_usd(
        self, amount: Decimal, from_cur: str, to_cur: str,
        rates: dict[str, Decimal],
    ) -> dict[str, Any]:
        """Convert fiat->fiat via USD pivot table."""
        usd_amount = self._to_usd(amount, from_cur, rates)
        if usd_amount is None:
            return {"error": f"no fiat rate for {from_cur}"}
        rate_to = rates.get(to_cur)
        if rate_to is None:
            return {"error": f"no fiat rate for {to_cur}"}
        result = usd_amount * rate_to
        rate = result / amount if amount > 0 else Decimal("0")
        return self._result(amount, from_cur, to_cur, rate)

    @staticmethod
    def _to_usd(
        amount: Decimal, currency: str, rates: dict[str, Decimal]
    ) -> Decimal | None:
        """Convert ``amount`` of ``currency`` to USD using rate table."""
        if currency == "USD":
            return amount
        rate = rates.get(currency)
        if rate is None or rate == 0:
            return None
        return amount / rate

    @staticmethod
    def _result(
        amount: Decimal, from_a: str, to_a: str, rate: Decimal
    ) -> dict[str, Any]:
        """Build compact result dict with float-rounded values."""
        result = amount * rate
        return {
            "amount": float(_round(amount, 8)),
            "from": from_a,
            "to": to_a,
            "rate": float(_round(rate, 8)),
            "result": float(_round(result, 8)),
        }

    async def close(self) -> None:
        """Close the shared HTTP session."""
        if self._session and not self._session.closed:
            await self._session.close()
