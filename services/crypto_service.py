"""Crypto & fiat currency conversion service.

Fetches live crypto prices from Binance public API (no auth) and fiat
exchange rates from ExchangeRate-API (open.er-api.com), with a fallback
to the Central Bank of Russia JSON wrapper (cbr-xml-daily.ru).

Also provides **historical** series for the trend tool:
    - ``get_crypto_history``  — OHLC klines from Binance.
    - ``get_fiat_history``    — daily rates from Frankfurter (ECB), with a
      fallback to the CBR historical archive for RUB and CIS currencies
      that ECB dropped (RUB, UAH, KZT, GEL, BYN, ...).

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
from datetime import datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation
from typing import TYPE_CHECKING, Any, Protocol

import aiohttp

from core.constants import (
    CBR_ARCHIVE_URL_TEMPLATE,
    CBR_DAILY_URL,
    CRYPTO_HISTORY_TTL,
    CRYPTO_PRICES_TTL,
    EXCHANGERATE_BASE_URL,
    FIAT_HISTORY_TTL,
    FIAT_RATES_TTL,
    FRANKFURTER_BASE_URL,
    REDIS_CRYPTO_HISTORY_KEY_PREFIX,
    REDIS_CRYPTO_KEY,
    REDIS_FIAT_HISTORY_KEY_PREFIX,
    REDIS_FIAT_KEY_PREFIX,
)
from core.exceptions import cupagentError

if TYPE_CHECKING:
    from redis.asyncio import Redis

logger = logging.getLogger(__name__)


class CryptoError(cupagentError):
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

    async def get_currency_history(
        self, from_asset: str, to_asset: str, *,
        days: int = ..., interval: str = ...,
    ) -> dict[str, Any]:
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

# Stablecoins pegged ~1:1 to USD. For *history* queries these are treated as
# fiat-equivalent (no Binance klines needed — just the USD→fiat daily
# series). They remain in _CRYPTO_ALIASES for the convert_currency tool
# (which uses Binance price endpoints and already handles them correctly).
_STABLECOINS: frozenset[str] = frozenset({"USDT", "USDC", "BUSD", "TUSD", "DAI"})

# Fiat currencies we know about (everything else is treated as crypto)
_KNOWN_FIAT: set[str] = {
    "USD", "EUR", "GBP", "RUB", "CNY", "UAH", "KZT", "GEL", "BYN",
    "TRY", "AED", "INR", "JPY", "KRW", "CHF", "CAD", "AUD", "PLN",
    "CZK", "SEK", "NOK", "DKK", "HKD", "SGD", "THB", "MYR", "PHP",
    "BRL", "MXN", "ZAR", "UZS", "KGS",
}

# Currencies supported by Frankfurter (ECB reference rates). ECB dropped
# RUB in March 2022 and never published UAH/KZT/GEL/BYN — those must go
# through the CBR archive instead.
_FRANKFURTER_FIAT: frozenset[str] = frozenset({
    "AUD", "BRL", "CAD", "CHF", "CNY", "CZK", "DKK", "EUR", "GBP",
    "HKD", "HUF", "IDR", "ILS", "INR", "ISK", "JPY", "KRW", "MXN",
    "MYR", "NOK", "NZD", "PHP", "PLN", "RON", "SEK", "SGD", "THB",
    "TRY", "USD", "ZAR",
})

# Binance kline intervals accepted by the history endpoint.
_CRYPTO_INTERVALS: dict[str, str] = {
    # user-facing alias → Binance interval code
    "5m": "5m", "1h": "1h", "1d": "1d",
}
# Per-interval max lookback in days (kept conservative to match the
# gift-history tool's limits).
_CRYPTO_INTERVAL_MAX_DAYS: dict[str, int] = {"5m": 31, "1h": 366, "1d": 1095}
DEFAULT_CRYPTO_INTERVAL = "1d"
DEFAULT_CRYPTO_HISTORY_DAYS = 7
DEFAULT_FIAT_HISTORY_DAYS = 30

# CBR archive fetch caps.  The archive is one JSON file per business day,
# so a 30-day window means up to 30 sequential HTTP requests — slow and
# rate-limit-prone.  We cap the lookback and the overall wall-clock time
# so the LLM tool call never blocks the chat for more than a few seconds.
_CBR_ARCHIVE_MAX_DAYS = 14       # cap lookback (vs. 365 for Frankfurter)
_CBR_ARCHIVE_TIMEOUT_S = 8.0     # hard cap per single archive fetch
_CBR_ARCHIVE_OVERALL_S = 12.0    # cap for the whole N-day fan-out
_CBR_ARCHIVE_CONCURRENCY = 4     # parallel archive fetches

# Overall wall-clock cap for a single currency-history tool call.  This
# wraps every provider path (Binance / Frankfurter / CBR) so a slow
# upstream can never block the LLM chat long enough to trip the upstream
# API's own read timeout.  If it fires, we return an ``error`` to the
# LLM rather than letting the request hang.
_HISTORY_OVERALL_TIMEOUT_S = 15.0


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
        frankfurter_base_url: str = "https://api.frankfurter.app",
    ) -> None:
        self._redis = redis
        self._binance_base = binance_base_url.rstrip("/")
        self._fiat_base = exchangerate_base_url.rstrip("/")
        self._cbr_url = cbr_daily_url
        self._frankfurter_base = frankfurter_base_url.rstrip("/")
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

    # ── History: high-level router ────────────────────────────────────

    async def get_currency_history(
        self,
        from_asset: str,
        to_asset: str,
        *,
        days: int = DEFAULT_CRYPTO_HISTORY_DAYS,
        interval: str = DEFAULT_CRYPTO_INTERVAL,
    ) -> dict[str, Any]:
        """Return an OHLC history of ``from_asset`` quoted in ``to_asset``.

        Routes automatically:
            - crypto → crypto/USD/USDT : Binance klines (``from`` vs USDT,
              ``to`` applied as a scalar ratio).
            - crypto → fiat            : Binance klines ``from``USDT,
              multiplied by USD→``to`` daily series.
            - fiat → fiat              : daily rate series (Frankfurter,
              fallback CBR archive).
            - fiat → crypto            : inverted crypto→fiat path.

        Each bar is ``{start (ISO), open, high, low, close}`` with float
        prices. ``source`` tells the LLM which provider answered.

        Args:
            from_asset: Source asset (alias or symbol).
            to_asset: Target (quote) asset.
            days: Lookback window in days (clamped per interval).
            interval: ``5m`` / ``1h`` / ``1d`` for crypto; fiat ignores
                this (always daily).

        Returns:
            Compact dict ``{from, to, source, interval, days, bars}``.
            On failure ``{from, to, error}``.
        """
        try:
            async with asyncio.timeout(_HISTORY_OVERALL_TIMEOUT_S):
                return await self._get_currency_history_impl(
                    from_asset, to_asset, days=days, interval=interval,
                )
        except asyncio.TimeoutError:
            logger.warning(
                "CryptoService: get_currency_history(%s→%s) timed out "
                "after %ss",
                from_asset, to_asset, _HISTORY_OVERALL_TIMEOUT_S,
            )
            return {
                "from": from_asset, "to": to_asset,
                "error": (
                    "история курса временно недоступна — внешний источник "
                    "не ответил вовремя, попробуй позже"
                ),
            }

    async def _get_currency_history_impl(
        self,
        from_asset: str,
        to_asset: str,
        *,
        days: int = DEFAULT_CRYPTO_HISTORY_DAYS,
        interval: str = DEFAULT_CRYPTO_INTERVAL,
    ) -> dict[str, Any]:
        """Implementation of :meth:`get_currency_history` (without timeout).

        Separated so :meth:`get_currency_history` can wrap the whole
        routing/fetch pipeline in a single ``asyncio.timeout`` without
        re-implementing the timeout in every provider path.
        """
        from_canon, from_kind = normalize_asset(from_asset)
        to_canon, to_kind = normalize_asset(to_asset)
        if from_kind == "unknown" or to_kind == "unknown":
            return {
                "from": from_asset, "to": to_asset,
                "error": f"unknown asset: {from_asset!r} or {to_asset!r}",
            }
        if from_canon == to_canon:
            return {
                "from": from_canon, "to": to_canon, "source": "none",
                "interval": interval, "days": days, "bars": [],
            }

        # Stablecoins (USDT/USDC/…) are ~1:1 USD — treat as fiat-equivalent
        # for history so we don't hit Binance klines for USDTUSDT (which
        # doesn't exist).  Both sides become "fiat-like" and we route
        # through the fiat daily-series path.
        from_is_stable = from_kind == "crypto" and from_canon in _STABLECOINS
        to_is_stable = to_kind == "crypto" and to_canon in _STABLECOINS

        if (from_kind == "crypto" or to_kind == "crypto") and not (
            from_is_stable or to_is_stable
        ):
            return await self._crypto_history(
                from_canon, from_kind, to_canon, to_kind,
                days=days, interval=interval,
            )

        # Both fiat, or one/both sides are stablecoins.  For stablecoins
        # we map them to USD and use the fiat series path.
        effective_from = "USD" if from_is_stable else from_canon
        effective_to = "USD" if to_is_stable else to_canon
        if effective_from != from_canon or effective_to != to_canon:
            # Remap the labels so the output still says USDT / USDC.
            result = await self.get_fiat_history(
                effective_from, effective_to, days=days,
            )
            # Replace USD labels back with the original stablecoin name
            # (keep "source" intact, only rename "from"/"to").
            result["from"] = from_canon
            result["to"] = to_canon
            # Rename bars if needed — but fiat bars use price values,
            # which are already correct (USD→fiat × 1).
            return result

        # Both fiat.
        return await self.get_fiat_history(
            from_canon, to_canon, days=days,
        )

    # ── History: crypto (Binance klines) ──────────────────────────────

    async def _crypto_history(
        self,
        from_canon: str, from_kind: str,
        to_canon: str, to_kind: str,
        *,
        days: int, interval: str,
    ) -> dict[str, Any]:
        """Fetch Binance klines for the crypto leg and rebase to ``to``.

        For crypto→crypto we fetch ``from``USDT and ``to``USDT and divide.
        For crypto↔fiat we fetch the crypto's USDT pair and multiply (or
        divide) by the USD→fiat rate series from Frankfurter/CBR.
        """
        iv = interval if interval in _CRYPTO_INTERVALS else DEFAULT_CRYPTO_INTERVAL
        max_days = _CRYPTO_INTERVAL_MAX_DAYS.get(iv, 1095)
        days = max(1, min(int(days or DEFAULT_CRYPTO_HISTORY_DAYS), max_days))

        # GRAM fallback to TON (1:1) when Binance has no GRAMUSDT.
        binance_from = from_canon if from_kind == "crypto" else to_canon
        gram_swap = binance_from == "GRAM"
        if gram_swap:
            binance_from = "TON"

        now = datetime.now(timezone.utc)
        start_ms = int((now - timedelta(days=days)).timestamp() * 1000)
        end_ms = int(now.timestamp() * 1000)

        # Crypto→crypto: need both pairs.
        if from_kind == "crypto" and to_kind == "crypto":
            bars_from = await self._fetch_binance_klines(
                f"{binance_from}USDT", iv, start_ms, end_ms,
            )
            to_sym = "TONUSDT" if to_canon == "GRAM" else f"{to_canon}USDT"
            bars_to = await self._fetch_binance_klines(
                to_sym, iv, start_ms, end_ms,
            )
            bars = _merge_pair_klines(bars_from, bars_to)
            source = "binance"
            title_from = from_canon
            title_to = to_canon
        else:
            bars_usd = await self._fetch_binance_klines(
                f"{binance_from}USDT", iv, start_ms, end_ms,
            )
            if to_kind == "fiat":
                # crypto → fiat: price_usd * usd_to_fiat
                rate_series = await self._fiat_rate_series("USD", to_canon, days)
                bars = _apply_rate_series(
                    bars_usd, rate_series, invert=False,
                )
                title_from = from_canon
                title_to = to_canon
            else:
                # fiat → crypto: invert (1 / price_usd) * usd_to_fiat
                rate_series = await self._fiat_rate_series("USD", from_canon, days)
                bars = _apply_rate_series(
                    bars_usd, rate_series, invert=True,
                )
                title_from = from_canon
                title_to = to_canon
            source = "binance+frankfurter/cbr"

        if not bars:
            return {
                "from": title_from,
                "to": title_to,
                "source": "none",
                "interval": iv,
                "days": days,
                "bars": [],
                "error": (
                    f"no data for {title_from}/{title_to} — "
                    "check that the pair exists on Binance"
                ),
            }

        return {
            "from": title_from,
            "to": title_to,
            "source": source,
            "interval": iv,
            "days": days,
            "bars": bars,
        }

    async def _fetch_binance_klines(
        self, symbol: str, interval: str, start_ms: int, end_ms: int,
    ) -> list[dict[str, Any]]:
        """Fetch and cache Binance klines as ``{start, open, high, low, close}``.

        Binance returns raw arrays ``[openTime, open, high, low, close, ...]``.
        We project to compact dicts with ISO-8601 ``start`` and float OHLC.
        """
        iv = _CRYPTO_INTERVALS.get(interval, interval)
        cache_key = (
            f"{REDIS_CRYPTO_HISTORY_KEY_PREFIX}:"
            f"{symbol}:{iv}:{start_ms}:{end_ms}"
        )
        cached = await self._redis.get(cache_key)
        if cached:
            try:
                return json.loads(cached)
            except json.JSONDecodeError:
                pass

        # Validate symbol is safe (ASCII alphanumeric) — Binance symbols
        # look like BTCUSDT, ETHUSDT.
        if not (isinstance(symbol, str) and symbol.isascii() and symbol.isalnum()):
            return []
        url = (
            f"{self._binance_base}/klines"
            f"?symbol={symbol}&interval={iv}"
            f"&startTime={start_ms}&endTime={end_ms}&limit=1000"
        )
        try:
            async with self.session.get(url) as resp:
                if resp.status != 200:
                    body = await resp.text()
                    logger.warning(
                        "CryptoService: Binance klines HTTP %d: %s",
                        resp.status, body[:200],
                    )
                    return []
                data = await resp.json()
        except aiohttp.ClientError as e:
            logger.warning("CryptoService: Binance klines error: %s", e)
            return []

        bars: list[dict[str, Any]] = []
        for row in data:
            if not isinstance(row, list) or len(row) < 5:
                continue
            try:
                bars.append({
                    "start": datetime.fromtimestamp(
                        int(row[0]) / 1000, tz=timezone.utc,
                    ).strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "open": float(row[1]),
                    "high": float(row[2]),
                    "low": float(row[3]),
                    "close": float(row[4]),
                })
            except (TypeError, ValueError):
                continue

        if bars:
            await self._redis.set(
                cache_key, json.dumps(bars), ex=CRYPTO_HISTORY_TTL,
            )
        return bars

    # ── History: fiat (Frankfurter, fallback CBR archive) ─────────────

    async def get_fiat_history(
        self,
        from_cur: str,
        to_cur: str,
        *,
        days: int = DEFAULT_FIAT_HISTORY_DAYS,
    ) -> dict[str, Any]:
        """Return a daily rate series for a fiat pair.

        Uses Frankfurter (ECB) when both currencies are in its reference
        set, otherwise falls back to the CBR historical archive
        (``cbr-xml-daily.ru/archive/YYYY/MM/DD/daily_json.js``), which is
        the only public source for RUB/UAH/KZT/GEL/BYN history.

        Args:
            from_cur: Canonical fiat code (e.g. ``"USD"``).
            to_cur: Canonical fiat code (e.g. ``"RUB"``).
            days: Lookback in days (clamped to 1..365).

        Returns:
            ``{from, to, source, interval: "1d", days, bars}`` where each
            bar is ``{start, open, high, low, close}`` (open=high=low=
            close because fiat has a single daily reference rate).
        """
        days = max(1, min(int(days or DEFAULT_FIAT_HISTORY_DAYS), 365))

        # Try Frankfurter first when both sides are ECB-supported.
        if from_cur in _FRANKFURTER_FIAT and to_cur in _FRANKFURTER_FIAT:
            bars = await self._fetch_frankfurter_series(from_cur, to_cur, days)
            if bars:
                return {
                    "from": from_cur, "to": to_cur, "source": "frankfurter",
                    "interval": "1d", "days": days, "bars": bars,
                }
            # Fall through to CBR if Frankfurter had nothing.

        # CBR archive path — works for RUB and any CBR-published currency.
        bars = await self._fetch_cbr_series(from_cur, to_cur, days)
        if bars:
            return {
                "from": from_cur, "to": to_cur, "source": "cbr",
                "interval": "1d", "days": days, "bars": bars,
            }
        return {
            "from": from_cur, "to": to_cur, "source": "none",
            "interval": "1d", "days": days, "bars": [],
        }

    async def _fiat_rate_series(
        self, base: str, quote: str, days: int,
    ) -> dict[str, Decimal]:
        """Return ``{date_iso: rate}`` for ``base`` quoted in ``quote``.

        Used to rebase crypto (USD-priced) klines into a fiat quote.
        Returns at most one rate per day; intraday crypto bars reuse the
        nearest preceding day's rate (caller handles via merge-on-date).
        """
        if base == quote:
            # Constant rate of 1 — caller still needs date keys, so emit
            # a single entry under a sentinel date.
            return {"__const__": Decimal("1")}
        series = await self._fetch_frankfurter_series(base, quote, days)
        if series:
            return {
                b["start"][:10]: Decimal(str(b["close"]))
                for b in series if b.get("close")
            }
        series = await self._fetch_cbr_series(base, quote, days)
        return {
            b["start"][:10]: Decimal(str(b["close"]))
            for b in series if b.get("close")
        }

    async def _fetch_frankfurter_series(
        self, base: str, quote: str, days: int,
    ) -> list[dict[str, Any]]:
        """Fetch a daily rate series from Frankfurter (time series endpoint).

        Frankfurter's ``/{start}..{end}?from=&to=`` returns
        ``{rates: {YYYY-MM-DD: {QUOTE: rate}}}``. We project each day to
        a ``{start, open, high, low, close}`` bar (fiat has one daily
        reference rate, so OHLC are identical).
        """
        cache_key = (
            f"{REDIS_FIAT_HISTORY_KEY_PREFIX}:frankfurter:{base}:{quote}:{days}"
        )
        cached = await self._redis.get(cache_key)
        if cached:
            try:
                return json.loads(cached)
            except json.JSONDecodeError:
                pass

        now = datetime.now(timezone.utc).date()
        start = now - timedelta(days=days)
        url = (
            f"{self._frankfurter_base}/"
            f"{start.isoformat()}..{now.isoformat()}"
            f"?from={base}&to={quote}"
        )
        try:
            async with self.session.get(url) as resp:
                if resp.status != 200:
                    logger.warning(
                        "CryptoService: Frankfurter HTTP %d", resp.status,
                    )
                    return []
                data = await resp.json()
        except aiohttp.ClientError as e:
            logger.warning("CryptoService: Frankfurter error: %s", e)
            return []

        rates = data.get("rates") if isinstance(data, dict) else None
        if not isinstance(rates, dict):
            return []

        bars: list[dict[str, Any]] = []
        for day in sorted(rates.keys()):
            entry = rates[day]
            if not isinstance(entry, dict):
                continue
            val = entry.get(quote)
            try:
                price = float(Decimal(str(val)))
            except (InvalidOperation, TypeError, ValueError):
                continue
            bars.append({
                "start": f"{day}T00:00:00Z",
                "open": price, "high": price,
                "low": price, "close": price,
            })

        if bars:
            await self._redis.set(
                cache_key, json.dumps(bars), ex=FIAT_HISTORY_TTL,
            )
        return bars

    async def _fetch_cbr_series(
        self, base: str, quote: str, days: int,
    ) -> list[dict[str, Any]]:
        """Build a fiat pair series from the CBR historical archive.

        CBR publishes per-day JSON at ``/archive/YYYY/MM/DD/daily_json.js``
        with per-unit-in-RUB rates. We sample the last ``days`` days
        (skipping weekends/holidays where no file exists), convert each
        currency to RUB, then derive the ``base``→``quote`` cross rate.

        Strategy: pick the currency that CBR lists directly (at least one
        of base/quote must be RUB or published by CBR). RUB is always the
        pivot.

        Performance caps (see constants at top of file): the lookback is
        clamped to ``_CBR_ARCHIVE_MAX_DAYS``, each individual archive
        fetch has a per-request timeout, and the whole fan-out is wrapped
        in an overall timeout — the CBR archive is a free service that
        can be slow/flaky, so we'd rather return fewer days than block
        the LLM chat for 60+ seconds.
        """
        # Clamp lookback — CBR archive is one HTTP call per day.
        effective_days = min(int(days or DEFAULT_FIAT_HISTORY_DAYS),
                             _CBR_ARCHIVE_MAX_DAYS)

        cache_key = (
            f"{REDIS_FIAT_HISTORY_KEY_PREFIX}:cbr:{base}:{quote}:{effective_days}"
        )
        cached = await self._redis.get(cache_key)
        if cached:
            try:
                return json.loads(cached)
            except json.JSONDecodeError:
                pass

        # Build the list of candidate dates (newest first).
        today = datetime.now(timezone.utc).date()
        dates = [(today - timedelta(days=i)) for i in range(effective_days)]
        # CBR publishes on Russian business days; we fetch a few recent
        # dates and keep whichever resolve. Limit concurrent requests to
        # avoid hammering the free service.
        semaphore = asyncio.Semaphore(_CBR_ARCHIVE_CONCURRENCY)

        async def _one(d) -> tuple[str, dict[str, Decimal] | None]:
            url = CBR_ARCHIVE_URL_TEMPLATE.format(
                year=d.year, month=d.month, day=d.day,
            )
            async with semaphore:
                try:
                    # Per-request timeout so a single hanging archive
                    # file doesn't stall the whole fan-out.
                    async with asyncio.timeout(_CBR_ARCHIVE_TIMEOUT_S):
                        async with self.session.get(url) as resp:
                            if resp.status != 200:
                                return d.isoformat(), None
                            data = await resp.json(content_type=None)
                except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                    logger.debug("CryptoService: CBR archive %s: %s", d, e)
                    return d.isoformat(), None
            return d.isoformat(), _cbr_to_rub_rates(data)

        # Overall cap: if the CBR archive is being slow/unresponsive,
        # bail out with whatever resolved so far instead of blocking
        # the LLM chat indefinitely.
        try:
            async with asyncio.timeout(_CBR_ARCHIVE_OVERALL_S):
                results = await asyncio.gather(*(_one(d) for d in dates))
        except asyncio.TimeoutError:
            logger.warning(
                "CryptoService: CBR archive fan-out timed out after "
                "%ss (%d/%d days requested)",
                _CBR_ARCHIVE_OVERALL_S, effective_days, days,
            )
            return []

        # Gather and keep only days that resolved. newest first.
        resolved = [(day, r) for day, r in results if r]
        if not resolved:
            return []
        resolved.sort(key=lambda x: x[0])

        bars: list[dict[str, Any]] = []
        for day, rub_rates in resolved:
            rate = _cross_from_rub(base, quote, rub_rates)
            if rate is None:
                continue
            price = float(rate)
            bars.append({
                "start": f"{day}T00:00:00Z",
                "open": price, "high": price,
                "low": price, "close": price,
            })

        if bars:
            await self._redis.set(
                cache_key, json.dumps(bars), ex=FIAT_HISTORY_TTL,
            )
        return bars

    async def close(self) -> None:
        """Close the shared HTTP session."""
        if self._session and not self._session.closed:
            await self._session.close()


# ── Module-level projection helpers for the history path ──────────────


def _cbr_to_rub_rates(data: Any) -> dict[str, Decimal] | None:
    """Extract ``{code: rub_per_unit}`` from a CBR daily JSON payload.

    CBR returns ``{Valute: {CODE: {Nominal, Value}}}`` where ``Value`` is
    RUB for ``Nominal`` units. We normalize to per-unit-in-RUB and add
    RUB itself at rate 1.
    """
    if not isinstance(data, dict):
        return None
    valute = data.get("Valute")
    if not isinstance(valute, dict):
        return None
    out: dict[str, Decimal] = {"RUB": Decimal("1")}
    for code, info in valute.items():
        if not isinstance(info, dict):
            continue
        try:
            nominal = Decimal(str(info.get("Nominal", 1)))
            value = Decimal(str(info["Value"]))
        except (InvalidOperation, KeyError, TypeError):
            continue
        if nominal > 0:
            out[code.upper()] = value / nominal
    return out


def _cross_from_rub(
    base: str, quote: str, rub_rates: dict[str, Decimal],
) -> Decimal | None:
    """Derive ``base``→``quote`` rate from a per-unit-in-RUB table.

    ``rub_rates`` maps currency → RUB for 1 unit. Cross rate
    base→quote = rub_per_base / rub_per_quote.
    """
    b = rub_rates.get(base)
    q = rub_rates.get(quote)
    if b is None or q is None or q == 0:
        return None
    return b / q


def _merge_pair_klines(
    bars_from: list[dict[str, Any]],
    bars_to: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Divide two aligned kline series to get from/to cross bars.

    Both inputs come from the same Binance interval/window so their
    timestamps line up. We match by ``start`` and divide OHLC element-
    wise; unmatched bars are dropped.
    """
    if not bars_from or not bars_to:
        return []
    by_start = {b["start"]: b for b in bars_to}
    out: list[dict[str, Any]] = []
    for bf in bars_from:
        bt = by_start.get(bf.get("start"))
        if not bt:
            continue
        try:
            out.append({
                "start": bf["start"],
                "open": bf["open"] / bt["open"],
                "high": bf["high"] / bt["high"],
                "low": bf["low"] / bt["low"],
                "close": bf["close"] / bt["close"],
            })
        except (TypeError, ZeroDivisionError):
            continue
    return out


def _apply_rate_series(
    bars_usd: list[dict[str, Any]],
    rate_series: dict[str, Decimal],
    *,
    invert: bool,
) -> list[dict[str, Any]]:
    """Rebase USD-priced klines by a daily fiat rate series.

    ``rate_series`` maps ``YYYY-MM-DD`` → ``Decimal`` (USD→quote or
    quote→USD depending on ``invert``). For crypto→fiat we multiply by
    USD→fiat; for fiat→crypto we invert the USD→crypto price and
    multiply by USD→fiat. The ``__const__`` sentinel (same currency) is
    treated as a flat rate of 1.

    Bars keep their original timestamp (intraday for 5m/1h).
    """
    if not bars_usd:
        return []
    # Sort the daily keys so we can pick the most-recent day at-or-before
    # each bar's date. Drop the sentinel key for the sort.
    day_keys = sorted(k for k in rate_series if k != "__const__")

    def _rate_for(start_iso: str) -> Decimal:
        if "__const__" in rate_series:
            return Decimal("1")
        day = start_iso[:10]
        # Most recent day at or before the bar's date.
        chosen: str | None = None
        for k in day_keys:
            if k <= day:
                chosen = k
            else:
                break
        if chosen is None and day_keys:
            chosen = day_keys[0]
        if chosen is None:
            return Decimal("1")
        return rate_series.get(chosen, Decimal("1"))

    out: list[dict[str, Any]] = []
    for b in bars_usd:
        start = b.get("start")
        r = _rate_for(start)
        if r == 0:
            continue
        try:
            if invert:
                # fiat → crypto: (1 / usd_price) * usd_to_fiat
                out.append({
                    "start": start,
                    "open": float((Decimal("1") / Decimal(str(b["open"]))) * r),
                    "high": float((Decimal("1") / Decimal(str(b["low"]))) * r),
                    "low": float((Decimal("1") / Decimal(str(b["high"]))) * r),
                    "close": float((Decimal("1") / Decimal(str(b["close"]))) * r),
                })
            else:
                # crypto → fiat: usd_price * usd_to_fiat
                out.append({
                    "start": start,
                    "open": float(Decimal(str(b["open"])) * r),
                    "high": float(Decimal(str(b["high"])) * r),
                    "low": float(Decimal(str(b["low"])) * r),
                    "close": float(Decimal(str(b["close"])) * r),
                })
        except (InvalidOperation, ZeroDivisionError, KeyError, TypeError):
            continue
    return out
