"""OpenRouter LLM client.

Sends user text to OpenRouter API and returns the model's response.
Default model: DeepSeek V4 Flash. Conversation history is kept per user
in Redis cache for short-term context.

Supports tool calling — the model can invoke any of the registered tools:

    - ``get_floor_prices``       — gift floor prices from marketplaces
    - ``get_collection_floors``  — per-model/backdrop floor prices
    - ``convert_currency``       — crypto/fiat conversion (Binance + CBR)
    - ``get_currency_history``   — crypto/fiat exchange-rate history
    - ``get_monochrome``         — GiftWiki monochrome classification

Tools are dynamically registered based on which services are available.
A callback hook allows the caller to provide UX feedback (e.g. status
message) when a tool is invoked.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Awaitable, Callable
from datetime import datetime, timedelta, timezone
from difflib import SequenceMatcher
from typing import Any, TYPE_CHECKING

import aiohttp

from core.exceptions import LLMError, ToolCallError

if TYPE_CHECKING:
    from services.crypto_service import CryptoService
    from services.gift_attrs_service import GiftAttrsService
    from services.giftwiki_service import GiftWikiService
    from services.moomin_service import MoominService
    from services.price_service import PriceService

logger = logging.getLogger(__name__)

# How many turns of history to keep per user
_MAX_HISTORY = 10
_HISTORY_TTL = 3600  # 1 hour

# Maximum tool-calling iterations (prevents infinite loops)
_MAX_TOOL_ROUNDS = 3

# Token limits — keep responses tight, reject oversized inputs.
_MAX_INPUT_CHARS = 2000      # ~500 tokens of Cyrillic/Latin user text
# Default output cap — enough for the bot's short reply style.
_DEFAULT_MAX_TOKENS = 1500

# How long to wait before retrying a rate-limited (429) request.
_RETRY_BACKOFF_BASE = 2     # seconds, doubled on each retry

# Fallback reply when the model produces no usable text after all rounds.
_EMPTY_REPLY_FALLBACK = "не получилось ответить, попробуй переформулировать"

# ── Tool schemas ─────────────────────────────────────────────────────────
# Compact descriptions to minimize tokens. Required fields kept empty
# when all params are optional.

_FLOOR_PRICES_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "get_floor_prices",
        "description": (
            "Получить актуальные floor prices с маркетплейсов. "
            "Может вернуть цены с одного маркета или со всех сразу, "
            "с фильтрацией по имени коллекции."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "market": {
                    "type": "string",
                    "enum": ["grapes", "mrkt", "portal", "getgems", "tonnel", "xgift"],
                    "description": (
                        "Маркетплейс. Если не указан — вернёт со всех."
                    ),
                },
                "query": {
                    "type": "string",
                    "description": (
                        "Английское слово/корень для поиска коллекции "
                        "(инструмент ищет подстроку в английских именах). "
                        "ПЕРЕВОДИ русский сленг на английский транслитом КАК "
                        "НАПИСАЛ ПОЛЬЗОВАТЕЛЬ, а не «как правильно»: лолпоп→"
                        "lolpop (НЕ lollipop!), пепе→pepe, змея→snake. "
                        "Названия коллекций часто нестандартные (LolPop, не "
                        "Lollipop). Результаты отсортированы по релевантности. "
                        "Никогда не шли русский — не найдёт."
                    ),
                },
                "top_n": {
                    "type": "integer",
                    "description": (
                        "Сколько коллекций вернуть с каждого маркета (по "
                        "умолчанию 20)."
                    ),
                },
            },
            "required": [],
        },
    },
}

_CONVERT_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "convert_currency",
        "description": (
            "Конвертировать крипту или фиат. Например: 5 GRAM в рублях, "
            "1 BTC в RUB, 100 долларов в рубли. Алиасы: btc/биткоин, "
            "ton/тон, gram/грам, eth, usdt, usd/доллар, rub/руб, eur/евро."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "amount": {
                    "type": "number",
                    "description": "Сколько конвертировать (например 5, 0.1, 1000).",
                },
                "from": {
                    "type": "string",
                    "description": "Исходная валюта/крипта (btc, gram, rub, usd...).",
                },
                "to": {
                    "type": "string",
                    "description": "Целевая валюта/крипта (rub, usd, gram...).",
                },
            },
            "required": ["amount", "from", "to"],
        },
    },
}

_MONOCHROME_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "get_monochrome",
        "description": (
            "Получить классификацию монохрома подарка: слабый/средний/"
            "сильный/комбо, либо НЕ монохром. Используй когда юзер "
            "спрашивает про монохром/тип подарка: по ссылке t.me/NAME-123, "
            "или по имени (кот 69, scared cat). Если юзер указал номер "
            "подарка (#3387, /scaredcat-3387) — обязательно передай number: "
            "тогда вернётся конкретный тип монохрома именно этого подарка. "
            "Без number вернёт все варианты монохрома для коллекции."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "gift_name": {
                    "type": "string",
                    "description": (
                        "Имя подарка в формате 'Имя Имя' (заглавная буква, "
                        "пробел между словами). НАПРИМЕР: 'Scared Cat', "
                        "'Plush Pepe', 'Lunar Snake', 'Santa Hat'. "
                        "Из ссылки t.me/scaredcat-69 → 'Scared Cat'. "
                        "Из 'кот 69' → 'Scared Cat'. НЕ используй slug "
                        "(scaredcat) или подчёркивания (scared_cat) — "
                        "только нормальное имя с пробелом."
                    ),
                },
                "number": {
                    "type": "integer",
                    "description": (
                        "Номер подарка внутри коллекции, если юзер его "
                        "указал: #3387, scaredcat-3387, «кот 3387». "
                        "Передав ЧИСЛО 3387 (не строку). По номеру "
                        "система определит точную модель и фон этого "
                        "подарка и вернёт один конкретный тип монохрома. "
                        "Если номера нет — не передавай поле."
                    ),
                },
            },
            "required": ["gift_name"],
        },
    },
}

_COLLECTION_ATTRIBUTES_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "get_collection_floors",
        "description": (
            "Получить floor prices моделей и фонов коллекции подарков. "
            "Используй когда юзер спрашивает цены моделей или фонов "
            "конкретной коллекции: «флор моделей scared cat», "
            "«сколько стоит фон Black у lunar snake», «какая модель "
            "дешевле у plush pepe». Цены в GRAM."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "collection_name": {
                    "type": "string",
                    "description": (
                        "Имя коллекции подарков (английское, с заглавной): "
                        "'Scared Cat', 'Plush Pepe', 'Lunar Snake', "
                        "'Santa Hat', 'Surge Board'."
                    ),
                },
            },
            "required": ["collection_name"],
        },
    },
}

_MARKET_SNAPSHOT_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "get_market_snapshot",
        "description": (
            "Актуальные кросс-маркет цены коллекции с агрегатора Moomin "
            "Market: цена на каждом маркете (grapes/mrkt/portals/getgems/"
            "tonnel/xgift) и прямой floor (минимальная из них). Используй "
            "когда юзер спрашивает «сколько сейчас стоит коллекция по "
            "всем маркетам» или хочет сравнить актуальные цены площадок. "
            "Цены в TON (= GRAM 1:1). Для одного маркета можно и "
            "get_floor_prices."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "collection": {
                    "type": "string",
                    "description": (
                        "Имя коллекции или слаг. Имя: 'Artisan Brick', "
                        "'Plush Pepe', 'Scared Cat'. Слаг тоже ок: "
                        "'artisanbrick'. Переводи сленг сам: кот→"
                        "Scared Cat, пепе→Plush Pepe, змея→Lunar Snake."
                    ),
                },
            },
            "required": ["collection"],
        },
    },
}

_PRICE_HISTORY_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "get_price_history",
        "description": (
            "История цены коллекции (OHLC-свечи) с агрегатора Moomin "
            "Market для одного маркета. Используй когда юзер спрашивает "
            "про тренд/динамику: «как менялась цена», «росла ли», "
            "«упала ли», «история цен», «что было неделю назад», "
            "«динамика за месяц». Вернёт сводку (open/high/low/close, "
            "% изменения, направление тренда) + ряд точек закрытия, "
            "НЕ сырой дамп свечей. Цены в TON (= GRAM 1:1)."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "collection": {
                    "type": "string",
                    "description": (
                        "Имя коллекции или слаг: 'Artisan Brick', "
                        "'plushpepe', 'Scared Cat'."
                    ),
                },
                "market": {
                    "type": "string",
                    "enum": ["grapes", "mrkt", "portals", "getgems",
                             "tonnel", "xgift"],
                    "description": (
                        "Маркет для истории. По умолчанию mrkt "
                        "(обычно самый ликвидный). Если юзер указал "
                        "маркет — передавай его."
                    ),
                },
                "interval": {
                    "type": "string",
                    "enum": ["5m", "1h", "1d"],
                    "description": (
                        "Свечи: 5m (до 31 дня), 1h (до 366 дней), "
                        "1d (до 1095 дней). По умолчанию 1d — для "
                        "динамики за неделю/месяц. 1h — для внутридневной, "
                        "5m — для короткой (минуты-часы)."
                    ),
                },
                "days": {
                    "type": "integer",
                    "description": (
                        "Сколько дней истории (от 1 до лимита интервала). "
                        "По умолчанию 7."
                    ),
                },
            },
            "required": ["collection"],
        },
    },
}

_CURRENCY_HISTORY_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "get_currency_history",
        "description": (
            "История курса криптовалюты или фиата: OHLC-свечи + сводка "
            "(open/high/low/close, % изменения, направление тренда) + "
            "ряд точек закрытия. Используй когда юзер спрашивает про "
            "динамику курса: «как менялся биток», «история курса тонкоина», "
            "«как рос доллар к рублю за месяц», «график евро», "
            "«что было с криптой за неделю». Источники: Binance (крипта), "
            "Frankfurter/ЕЦБ (USD/EUR/GBP/CNY...), ЦБ РФ (рубль/гривна/"
            "тенге/лари/белруб). Покрывает fiat→fiat, crypto→crypto, "
            "crypto→fiat и наоборот. Алиасы те же что у convert_currency."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "from": {
                    "type": "string",
                    "description": (
                        "Исходная валюта/крипта: btc, gram, ton, eth, "
                        "usd, rub, eur и т.д. Сленг ок: биток→btc, "
                        "грам→gram, руб→rub."
                    ),
                },
                "to": {
                    "type": "string",
                    "description": (
                        "Целевая валюта/крипта для котировки. Например: "
                        "from=btc to=rub → история битка в рублях. "
                        "from=usd to=rub → история курса доллара."
                    ),
                },
                "interval": {
                    "type": "string",
                    "enum": ["5m", "1h", "1d"],
                    "description": (
                        "Свечи для крипты: 5m/1h/1d (по умолчанию 1d). "
                        "Фиат всегда дневной (поле игнорируется)."
                    ),
                },
                "days": {
                    "type": "integer",
                    "description": (
                        "Сколько дней истории (от 1 до лимита интервала). "
                        "По умолчанию 7 для крипты, 30 для фиата."
                    ),
                },
            },
            "required": ["from", "to"],
        },
    },
}


def _build_tools(services: dict[str, Any]) -> list[dict[str, Any]]:
    """Build the tool list based on which services are available.

    A service that is None or missing causes its tool(s) to be omitted,
    so the model never calls a tool that can't be served.
    """
    tools: list[dict[str, Any]] = []
    if services.get("price_service") is not None:
        tools.append(_FLOOR_PRICES_TOOL)
        tools.append(_COLLECTION_ATTRIBUTES_TOOL)
    if services.get("crypto_service") is not None:
        tools.append(_CONVERT_TOOL)
        tools.append(_CURRENCY_HISTORY_TOOL)
    if services.get("giftwiki_service") is not None:
        tools.append(_MONOCHROME_TOOL)
    if services.get("moomin_service") is not None:
        tools.append(_MARKET_SNAPSHOT_TOOL)
        tools.append(_PRICE_HISTORY_TOOL)
    return tools


# Registry of known tool names (used for filtering model output).
_KNOWN_TOOL_NAMES: frozenset[str] = frozenset({
    "get_floor_prices",
    "get_collection_floors",
    "convert_currency",
    "get_currency_history",
    "get_monochrome",
    "get_market_snapshot",
    "get_price_history",
})


class LLMService:
    """Talks to OpenRouter to produce chat responses.

    Supports a **fallback model chain**: if the primary model returns 429
    (rate-limit) or 5xx, the service automatically tries the next model in
    the chain (with exponential backoff). Free models (``:free`` suffix)
    are rate-limited harder on OpenRouter, so the chain retries with delays.
    """

    def __init__(
        self,
        *,
        api_key: str,
        model: str = "deepseek/deepseek-v4-flash",
        base_url: str = "https://openrouter.ai/api/v1",
        system_prompt: str = "",
        max_tokens: int = _DEFAULT_MAX_TOKENS,
        fallback_models: list[str] | None = None,
        skills: dict[str, str] | None = None,
    ) -> None:
        """Initialize the LLM service.

        Args:
            api_key: OpenRouter API key.
            model: Primary model id (e.g. ``deepseek/deepseek-v4-flash``).
            base_url: OpenRouter API base URL.
            system_prompt: The base system prompt (instruction.md). Always
                sent with every request.
            max_tokens: Max output tokens per reply.
            fallback_models: Models to try on 429/5xx, in order.
            skills: Optional mapping of skill name → skill markdown body.
                Skills are NOT sent by default; they are appended to the
                system prompt lazily, per-request, based on keyword
                matching in :meth:`_build_system_prompt`. This keeps the
                prompt small (~11 KB of skills only attached when the
                user's message is relevant to that skill).
        """
        self._api_key = api_key
        self._model = model
        self._base_url = base_url.rstrip("/")
        self._system_prompt = system_prompt
        self._max_tokens = max_tokens
        self._skills: dict[str, str] = dict(skills) if skills else {}
        # Model chain: primary first, then fallbacks (deduplicated).
        chain = [model, *(fallback_models or [])]
        seen: set[str] = set()
        self._model_chain: list[str] = []
        for m in chain:
            if m.strip() not in seen:
                seen.add(m.strip())
                self._model_chain.append(m.strip())
        self._session: aiohttp.ClientSession | None = None

    # Keyword triggers per skill. A skill is attached when the user's
    # message contains any of its trigger words (case-insensitive). The
    # lists are kept tight to avoid false positives — when in doubt, the
    # base system_prompt alone is enough (the LLM already knows which tool
    # to call from instruction.md). Each skill is roughly 2-3 KB, so
    # skipping an irrelevant skill saves ~500-800 input tokens per request.
    _SKILL_TRIGGERS: dict[str, tuple[str, ...]] = {
        "market-prices": (
            "флор", "floor", "цена", "цене", "цены", "сколько стоит",
            "выгоднее", "дешевле", "дороге", "подешевле", "по курсу",
            "грам", "gram", "ton", "тон", "крипт", "курс", "маркет",
            "грейпс", "мркт", "порталс", "гетгемс", "тоннел", "хгифт",
            "grapes", "mrkt", "portal", "getgems", "tonnel", "xgift",
            "коллекция стоит", "стоит подарок",
        ),
        "slang": (
            "кто такой", "кто создал", "что такое", "что за",
            "автогифтс", "autogifts", " p2p ", "p2p-", "скам", "scam",
            "казино", "rolls", "balls", "гифтсдабл", "giftsdouble",
            "вкладывать", "что купить", "перспективн", "рейд", "дроп",
            "саплай", "supply", "холдер", "флип", "lore", "лор",
        ),
        "gift-numbers": (
            "тир", "tier", "ценный номер", "ценность номера", "ценный",
            "зеркальн", "палиндром", "фибоначчи", "удачный номер",
            "какой номер цен", "номер цен",
        ),
        "orig-accounts": (
            "ориг", "оригинал", "релеер", "relayer", "настоящий аккаунт",
            "оригинальный аккаунт", "это ориг", "original",
        ),
        "market-history": (
            "тренд", "динамика", "история цен", "как менялась",
            "росла", "упала", "выросла", "снизилась", "изменилась",
            "за неделю", "за месяц", "за день", "за последние",
            "что было", "раньше стоил", "раньше стоила",
            "график", "chart", "trend", "history", "grew", "fell",
            "risen", "dropped", "performance",
        ),
        "currency-history": (
            "как менялся курс", "история курса", "курс рос", "курс упал",
            "курс росла", "курс вырос", "динамика курса", "график курса",
            "курс за неделю", "курс за месяц", "курс за день",
            "как рос биток", "как падал биток", "биток рос", "биток падал",
            "тонкоин рос", "грам рос", "эфир рос", "как менялся биток",
            "как менялся тон", "как менялся грам", "как менялся эфир",
            "доллар рос", "доллар падал", "евро рос", "евро падал",
            "рубль падал", "рубль креп", "как рос доллар", "как рос евро",
            "история битка", "история тонкоина", "история эфира",
            "история доллара", "история евро", "история рубля",
            "что было с битком", "что было с тонкоином", "что было с курсом",
            "btc trend", "eth trend", "ton trend", "gram trend",
        ),
    }

    def _build_system_prompt(self, user_text: str) -> str:
        """Assemble the system prompt for a given user message.

        Always starts with the base ``instruction.md`` content, then
        appends any skill whose trigger keywords appear in ``user_text``.
        This keeps the prompt minimal — most price/currency questions
        don't need the slang, gift-numbers, or orig-accounts skills, so
        those ~6 KB of markdown are skipped entirely.

        Args:
            user_text: The user's current message (already lowercased
                for matching is handled internally).

        Returns:
            The assembled system prompt string.
        """
        prompt = self._system_prompt
        if not self._skills:
            return prompt
        haystack = user_text.lower()
        for skill_name, triggers in self._SKILL_TRIGGERS.items():
            body = self._skills.get(skill_name)
            if not body:
                continue
            if any(t in haystack for t in triggers):
                prompt += f"\n\n# Skill: {skill_name}\n\n{body}"
        return prompt

    @property
    def session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "Content-Type": "application/json",
                    "HTTP-Referer": "https://github.com/cupagent",
                    "X-Title": "cupagent",
                },
                timeout=aiohttp.ClientTimeout(total=60),
            )
        return self._session

    async def chat(
        self,
        user_text: str,
        *,
        user_id: int,
        history: list[dict[str, str]] | None = None,
        price_service: PriceService | None = None,
        crypto_service: CryptoService | None = None,
        giftwiki_service: GiftWikiService | None = None,
        gift_attrs_service: GiftAttrsService | None = None,
        moomin_service: MoominService | None = None,
        on_tool_call: Callable[[], Awaitable[None]] | None = None,
    ) -> str:
        """Send a message to the LLM and return the response text.

        If the model invokes a tool, the result is appended and the model
        is called again — up to ``_MAX_TOOL_ROUNDS`` iterations.

        Args:
            user_text: The user's message.
            user_id: Telegram user ID (for logging / future tracking).
            history: Optional prior conversation turns.
            price_service: Optional PriceService for floor-price lookups.
            crypto_service: Optional CryptoService for crypto/fiat conversion.
            giftwiki_service: Optional GiftWikiService for collection lookup.
            gift_attrs_service: Optional GiftAttrsService for resolving a
                specific gift number to its model/backdrop.
            moomin_service: Optional MoominService for cross-market
                snapshots and OHLC price history.
            on_tool_call: Optional async callback invoked before executing
                a tool (e.g. to update UI status).

        Returns:
            Model's reply text.

        Raises:
            ToolCallError: If the model invokes a tool but the required
                service is not available.
            LLMError: On API or network errors.
        """
        services: dict[str, Any] = {
            "price_service": price_service,
            "crypto_service": crypto_service,
            "giftwiki_service": giftwiki_service,
            "gift_attrs_service": gift_attrs_service,
            "moomin_service": moomin_service,
        }
        tools = _build_tools(services)

        # Hard cap on input length — protects against oversized prompts
        # (paste dumps, spam, prompt-injection payloads).
        truncated_text = user_text[:_MAX_INPUT_CHARS]

        # Build a per-request system prompt: base instruction + any skills
        # whose trigger keywords appear in this user message.
        system_prompt = self._build_system_prompt(user_text)

        messages: list[dict[str, Any]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        if history:
            messages.extend(history[-_MAX_HISTORY:])
        messages.append({"role": "user", "content": truncated_text})

        payload: dict[str, Any] = self._base_payload(messages)
        if tools:
            payload["tools"] = tools

        reply_text = ""
        for round_idx in range(_MAX_TOOL_ROUNDS + 1):
            reply_text, tool_calls = await self._call_api(payload, user_id)

            if not tool_calls:
                if not reply_text.strip():
                    logger.warning(
                        "LLMService: empty reply for user %d after round %d "
                        "(%d messages in context)",
                        user_id, round_idx, len(messages),
                    )
                    return _EMPTY_REPLY_FALLBACK
                return reply_text.strip()

            # Filter to known tool names we can actually execute.
            valid_calls: list[dict[str, Any]] = []
            for tc in tool_calls:
                fn_name = tc.get("function", {}).get("name", "")
                if fn_name not in _KNOWN_TOOL_NAMES:
                    logger.warning(
                        "LLMService: unknown tool call %s, skipping", fn_name
                    )
                    continue
                if not _service_supports(fn_name, services):
                    logger.warning(
                        "LLMService: tool %s requested but service missing",
                        fn_name,
                    )
                    continue
                valid_calls.append(tc)

            if not valid_calls:
                if not reply_text.strip():
                    return _EMPTY_REPLY_FALLBACK
                return reply_text.strip()

            # Notify caller (typing preview update) once.
            if on_tool_call is not None:
                try:
                    await on_tool_call()
                except Exception:
                    logger.warning(
                        "LLMService: on_tool_call callback failed",
                        exc_info=True,
                    )

            # Execute all tool calls concurrently.
            results = await asyncio.gather(
                *(self._execute_tool_call(services, tc) for tc in valid_calls),
                return_exceptions=True,
            )

            # Append ONE assistant message carrying all tool_calls
            # (API requires role=assistant with tool_calls array).
            assistant_msg: dict[str, Any] = {
                "role": "assistant",
                "content": reply_text or None,
                "tool_calls": [
                    {
                        "id": tc.get("id", ""),
                        "type": "function",
                        "function": tc.get("function", {}),
                    }
                    for tc in valid_calls
                ],
            }
            messages.append(assistant_msg)

            # Append each tool result as a separate role=tool message.
            for tc, result in zip(valid_calls, results):
                content = (
                    result if isinstance(result, str)
                    else json.dumps(
                        {"error": f"{type(result).__name__}: {result}"},
                        ensure_ascii=False,
                    )
                )
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.get("id", ""),
                    "content": content,
                })

            # Re-send with tool results (payload already has tools).
            payload = self._base_payload(messages)
            payload["tools"] = tools

        # If we exhausted rounds, make one final call with tool_choice="none"
        # (forces the model to answer in text, not call another tool).
        # Some models still emit tool_calls even without `tools` in the
        # payload — tool_choice="none" is the only reliable way to forbid it.
        logger.warning(
            "LLMService: exhausted %d tool-calling rounds for user %d, "
            "forcing final reply with tool_choice=none",
            _MAX_TOOL_ROUNDS, user_id,
        )
        final_payload = self._base_payload(messages)
        final_payload["tool_choice"] = "none"
        final_reply, _ = await self._call_api(final_payload, user_id)
        final_reply = final_reply.strip()
        if final_reply:
            return final_reply

        # Last-resort fallback: the model refused to produce any text even
        # with tool_choice="none". Surface a short notice rather than an
        # empty message (which would render as just the ghost footer).
        logger.warning(
            "LLMService: empty reply after fallback for user %d", user_id,
        )
        return _EMPTY_REPLY_FALLBACK

    async def _execute_tool_call(
        self, services: dict[str, Any], tool_call: dict[str, Any]
    ) -> str:
        """Parse args and route a tool call to the right service handler.

        Args:
            services: Mapping of service name -> instance.
            tool_call: The tool call dict from the model.

        Returns:
            JSON string with the tool result (or an error dict).
        """
        fn_args_str = tool_call.get("function", {}).get("arguments", "")
        fn_args: dict[str, Any] = {}
        try:
            fn_args = json.loads(fn_args_str)
        except (json.JSONDecodeError, TypeError):
            logger.warning(
                "LLMService: failed to parse tool args: %s", fn_args_str
            )
            return json.dumps(
                {"error": f"invalid arguments: {fn_args_str}"},
                ensure_ascii=False,
            )

        fn_name = tool_call.get("function", {}).get("name", "")
        try:
            if fn_name == "get_floor_prices":
                return await self._tool_floor_prices(
                    services["price_service"], fn_args,
                )
            if fn_name == "get_collection_floors":
                return await self._tool_get_collection_floors(
                    services["price_service"], fn_args,
                )
            if fn_name == "convert_currency":
                return await self._tool_convert_currency(
                    services["crypto_service"], fn_args,
                )
            if fn_name == "get_currency_history":
                return await self._tool_currency_history(
                    services["crypto_service"], fn_args,
                )
            if fn_name == "get_monochrome":
                return await self._tool_get_monochrome(
                    services["giftwiki_service"], fn_args,
                    services.get("gift_attrs_service"),
                )
            if fn_name == "get_market_snapshot":
                return await self._tool_market_snapshot(
                    services["moomin_service"], fn_args,
                )
            if fn_name == "get_price_history":
                return await self._tool_price_history(
                    services["moomin_service"], fn_args,
                )
            return json.dumps(
                {"error": f"unknown tool: {fn_name}"}, ensure_ascii=False,
            )
        except Exception as e:  # noqa: BLE001 — surface to LLM
            logger.exception("LLMService: tool %s failed", fn_name)
            return json.dumps(
                {"error": f"{type(e).__name__}: {e}"}, ensure_ascii=False,
            )

    def _base_payload(
        self, messages: list[dict[str, Any]], *, model: str | None = None,
    ) -> dict[str, Any]:
        """Build the common request payload shared by all API calls.

        Args:
            messages: Chat message history.
            model: Override model name (used by the fallback chain).
        """
        return {
            "model": model or self._model,
            "messages": messages,
            "temperature": 0,
            "max_tokens": self._max_tokens,
        }

    async def _call_api(
        self, payload: dict[str, Any], user_id: int,
    ) -> tuple[str, list[dict[str, Any]]]:
        """Call the OpenRouter API with fallback chain + retry.

        On 429 (rate-limit) or 5xx, tries the next model in the chain
        with exponential backoff. Returns ``(reply_text, tool_calls)`` on
        success, or raises ``LLMError`` when all models are exhausted.
        """
        last_error: str | None = None
        backoff = _RETRY_BACKOFF_BASE

        for model_name in self._model_chain:
            payload["model"] = model_name

            for attempt in range(3):  # up to 3 retries per model
                try:
                    async with self.session.post(
                        f"{self._base_url}/chat/completions",
                        json=payload,
                    ) as resp:
                        if resp.status == 429:
                            last_error = f"429 rate-limited on {model_name}"
                            logger.warning(
                                "LLMService: %s (attempt %d, retry in %ds)",
                                last_error, attempt + 1, backoff,
                            )
                            await asyncio.sleep(backoff)
                            backoff = min(backoff * 2, 30)
                            continue

                        if resp.status >= 500:
                            last_error = (
                                f"HTTP {resp.status} on {model_name}"
                            )
                            logger.warning(
                                "LLMService: %s (attempt %d)",
                                last_error, attempt + 1,
                            )
                            await asyncio.sleep(backoff)
                            backoff = min(backoff * 2, 30)
                            continue

                        if resp.status != 200:
                            body = await resp.text()
                            logger.error(
                                "LLMService: OpenRouter error %d on %s: %s",
                                resp.status, model_name, body[:500],
                            )
                            raise LLMError(
                                f"OpenRouter returned HTTP {resp.status} "
                                f"({model_name}): {body[:200]}"
                            )

                        data = await resp.json()

                        choice = data.get("choices", [{}])[0]
                        message = choice.get("message", {})
                        reply = message.get("content", "") or ""
                        tool_calls = message.get("tool_calls") or []

                        if not reply and not tool_calls:
                            logger.warning(
                                "LLMService: empty response from %s: %s",
                                model_name,
                                json.dumps(message, ensure_ascii=False)[:500],
                            )

                        logger.info(
                            "LLMService: reply for user %d via %s "
                            "(%d chars, %d tool_calls)",
                            user_id, model_name, len(reply), len(tool_calls),
                        )
                        return reply, tool_calls

                except aiohttp.ClientError as e:
                    last_error = f"network error on {model_name}: {e}"
                    logger.warning(
                        "LLMService: %s (attempt %d)",
                        last_error, attempt + 1,
                    )
                    await asyncio.sleep(backoff)
                    backoff = min(backoff * 2, 30)
                except asyncio.TimeoutError:
                    last_error = f"timeout on {model_name}"
                    logger.warning(
                        "LLMService: %s (attempt %d)",
                        last_error, attempt + 1,
                    )
                    await asyncio.sleep(backoff)
                    backoff = min(backoff * 2, 30)

            # All retries failed for this model — try next in chain.
            backoff = _RETRY_BACKOFF_BASE

        raise LLMError(
            f"All models exhausted: {last_error or 'no response'}"
        )

    # ── Tool implementations ──────────────────────────────────────────

    async def _tool_floor_prices(
        self, price_service: PriceService, args: dict[str, Any],
    ) -> str:
        """Execute get_floor_prices and return compact JSON."""
        from core.types import MarketName

        market_map: dict[str, MarketName] = {
            "grapes": MarketName.GRAPES,
            "mrkt": MarketName.MRKT,
            "portal": MarketName.PORTAL,
            "getgems": MarketName.GETGEMS,
            "tonnel": MarketName.TONNEL,
            "xgift": MarketName.XGIFT,
        }
        market = args.get("market")
        query = args.get("query")
        top_n = args.get("top_n")

        if market:
            m = market_map.get(market.lower())
            if m is None:
                return json.dumps(
                    {"error": f"Unknown market: {market}"},
                    ensure_ascii=False,
                )
            # Check if this market is actually configured in price_service
            if m not in price_service._clients:
                return json.dumps(
                    {"market": m.value, "error": "market not configured"},
                    ensure_ascii=False,
                )
            targets = [m]
        else:
            # Only include markets that are actually configured
            targets = [m for m in market_map.values() if m in price_service._clients]

        query_norm = _normalize_query(query) if query else None
        cap = int(top_n) if isinstance(top_n, int) and top_n > 0 else 20

        fetched = await asyncio.gather(
            *(price_service.get_floor_prices(m) for m in targets),
            return_exceptions=True,
        )

        result_markets: list[dict[str, Any]] = []
        for m, raw in zip(targets, fetched):
            if isinstance(raw, Exception):
                result_markets.append({
                    "market": m.value,
                    "error": f"{type(raw).__name__}: {raw}",
                })
                continue

            matches = _filter_and_pick(raw, query_norm, cap)
            # Filter out entries with null price — LLM will see them as
            # "null" and produce unhelpful responses.
            priced = [p for p in matches if p.price is not None]
            if not priced:
                # All prices are null for this market — include it with
                # empty prices list so the LLM can say "no listings".
                result_markets.append({
                    "market": m.value,
                    "total": len(raw),
                    "returned": 0,
                    "prices": [],
                    "all_null": True,
                })
                continue
            result_markets.append({
                "market": m.value,
                "total": len(raw),
                "returned": len(priced),
                "prices": [
                    {"name": p.gift_name, "price": p.price}
                    for p in priced
                ],
            })

        return json.dumps(
            {"query": query, "markets": result_markets},
            ensure_ascii=False,
        )

    async def _tool_get_collection_floors(
        self, price_service: PriceService, args: dict[str, Any],
    ) -> str:
        """Execute get_collection_floors and return compact JSON.

        Returns per-market model + backdrop floor prices for a collection.
        Markets that don't support attribute floors or returned nothing
        are omitted — the LLM is instructed to skip missing markets.
        """
        collection_name = args.get("collection_name", "")
        result = await price_service.get_collection_attributes(collection_name)
        return json.dumps(
            {
                "collection": collection_name,
                "markets": result,
            },
            ensure_ascii=False,
        )

    async def _tool_market_snapshot(
        self, moomin_service: MoominService, args: dict[str, Any],
    ) -> str:
        """Execute get_market_snapshot and return compact JSON.

        Returns per-market current prices + direct floor. Markets with a
        null/missing price are dropped so the LLM never prints "None".
        A 404 (unknown collection) is surfaced as an explicit error dict
        so the model can say the collection wasn't found.
        """
        collection = args.get("collection", "")
        try:
            snap = await moomin_service.get_snapshot(collection)
        except Exception as e:  # noqa: BLE001 — surface to LLM
            logger.warning("LLMService: get_market_snapshot failed: %s", e)
            return json.dumps(
                {"collection": collection, "error": f"{type(e).__name__}: {e}"},
                ensure_ascii=False,
            )
        if snap is None:
            return json.dumps(
                {
                    "collection": collection,
                    "error": "коллекция не найдена. проверь имя или попробуй синоним",
                },
                ensure_ascii=False,
            )
        priced = [p for p in snap.get("prices", []) if p.get("price")]
        return json.dumps(
            {
                "collection": snap.get("title") or collection,
                "slug": snap.get("slug"),
                "quote_asset": snap.get("quote_asset", "TON"),
                "prices": priced,
                "direct_floor": snap.get("direct_floor"),
                "note": (
                    "цены в TON (= GRAM 1:1). покажи по маркетам и "
                    "отметь где дешевле (direct_floor)."
                ),
            },
            ensure_ascii=False,
        )

    async def _tool_price_history(
        self, moomin_service: MoominService, args: dict[str, Any],
    ) -> str:
        """Execute get_price_history and return a trend summary JSON.

        The Moomin candles endpoint can return hundreds of bars — that
        would blow the LLM token budget. Instead of dumping them raw,
        this computes a compact summary (open/high/low/close, % change,
        direction) plus a downsampled series of ~12 close points so the
        model can describe the shape of the trend.
        """
        collection = args.get("collection", "")
        market = args.get("market") or "mrkt"
        interval = args.get("interval") or "1d"
        days = self._coerce_int(args.get("days")) or _DEFAULT_HISTORY_DAYS

        # Clamp days to the interval's max lookback.
        max_days = _INTERVAL_MAX_DAYS.get(interval, _DEFAULT_MAX_DAYS)
        days = max(1, min(days, max_days))

        now = datetime.now(timezone.utc)
        from_dt = (now - timedelta(days=days)).strftime("%Y-%m-%dT%H:%M:%SZ")
        to_dt = now.strftime("%Y-%m-%dT%H:%M:%SZ")

        try:
            data = await moomin_service.get_candles(
                collection,
                market=market,
                interval=interval,
                from_dt=from_dt,
                to_dt=to_dt,
            )
        except Exception as e:  # noqa: BLE001 — surface to LLM
            logger.warning("LLMService: get_price_history failed: %s", e)
            return json.dumps(
                {"collection": collection, "error": f"{type(e).__name__}: {e}"},
                ensure_ascii=False,
            )
        if data is None:
            return json.dumps(
                {
                    "collection": collection,
                    "error": "коллекция не найдена. проверь имя или попробуй синоним",
                },
                ensure_ascii=False,
            )
        bars = data.get("candles", [])
        summary = _summarize_candles(bars)
        series = _downsample_close_series(bars, _HISTORY_SERIES_POINTS)
        return json.dumps(
            {
                "collection": data.get("title") or collection,
                "slug": data.get("slug"),
                "market": data.get("market", market),
                "interval": data.get("interval", interval),
                "period_days": days,
                "bars": len(bars),
                "quote_asset": data.get("quote_asset", "TON"),
                "summary": summary,
                "series": series,
                "note": (
                    "цены в TON (= GRAM 1:1). series — прореженный ряд "
                    "точек закрытия [{t, close}]. опиши тренд по summary "
                    "(direction + change_pct) и при необходимости форму "
                    "по series."
                ),
            },
            ensure_ascii=False,
        )

    async def _tool_convert_currency(
        self, crypto_service: CryptoService, args: dict[str, Any],
    ) -> str:
        """Execute convert_currency and return compact JSON."""
        result = await crypto_service.convert(
            amount=args.get("amount", 0),
            from_asset=args.get("from", ""),
            to_asset=args.get("to", ""),
        )
        return json.dumps(result, ensure_ascii=False)

    async def _tool_currency_history(
        self, crypto_service: CryptoService, args: dict[str, Any],
    ) -> str:
        """Execute get_currency_history and return a trend summary JSON.

        Mirrors :meth:`_tool_price_history`: the underlying service can
        return hundreds of bars (Binance 1h over a year), so we summarize
        into ``summary`` (open/high/low/close, % change, direction) plus
        a downsampled close ``series`` of ~12 points. The model describes
        the trend from the summary and, if useful, the series shape.
        """
        from_asset = args.get("from", "")
        to_asset = args.get("to", "")
        interval = args.get("interval") or "1d"
        # Default days: 7 for crypto pairs, 30 for fiat-only.
        days_raw = self._coerce_int(args.get("days"))

        # Determine asset kinds to pick a sensible default lookback.
        try:
            from services.crypto_service import normalize_asset
            _, from_kind = normalize_asset(from_asset)
            _, to_kind = normalize_asset(to_asset)
        except Exception:  # noqa: BLE001 — defensive, fall back to crypto default
            from_kind = to_kind = "crypto"
        fiat_only = from_kind == "fiat" and to_kind == "fiat"

        if days_raw is None:
            days = 30 if fiat_only else _DEFAULT_HISTORY_DAYS
        else:
            days = days_raw

        try:
            data = await crypto_service.get_currency_history(
                from_asset, to_asset, days=days, interval=interval,
            )
        except Exception as e:  # noqa: BLE001 — surface to LLM
            logger.warning("LLMService: get_currency_history failed: %s", e)
            return json.dumps(
                {"from": from_asset, "to": to_asset,
                 "error": f"{type(e).__name__}: {e}"},
                ensure_ascii=False,
            )
        if "error" in data:
            return json.dumps(data, ensure_ascii=False)

        bars = data.get("bars", [])
        summary = _summarize_candles(bars)
        series = _downsample_close_series(bars, _HISTORY_SERIES_POINTS)
        return json.dumps(
            {
                "from": data.get("from", from_asset),
                "to": data.get("to", to_asset),
                "source": data.get("source"),
                "interval": data.get("interval", interval),
                "period_days": data.get("days", days),
                "bars": len(bars),
                "summary": summary,
                "series": series,
                "note": (
                    "series — прореженный ряд точек закрытия [{t, close}]. "
                    "опиши тренд по summary (direction + change_pct) и при "
                    "необходимости форму по series. валюты пиши по правилам "
                    "инструкции: крипта тикером (BTC, GRAM), фиат словами "
                    "(рублей, долларов, евро)."
                ),
            },
            ensure_ascii=False,
        )

    async def _tool_get_monochrome(
        self,
        giftwiki_service: GiftWikiService,
        args: dict[str, Any],
        gift_attrs_service: GiftAttrsService | None = None,
    ) -> str:
        """Execute get_monochrome and return compact JSON.

        If ``number`` is given and a ``GiftAttrsService`` is available,
        we resolve the specific instance's model + backdrop from its
        t.me preview page and filter the collection matrix down to the
        single applicable monochrome ``type`` — so the model can answer
        "which monochrome is #3387?" instead of listing every type.
        """
        gift_name = args.get("gift_name", "")
        number_raw = args.get("number")
        number = self._coerce_int(number_raw)

        results = await giftwiki_service.get_monochromes(gift_name=gift_name)

        # Resolve the specific instance by number → model + backdrop →
        # narrow the collection matrix to the one real type.
        if number is not None and gift_attrs_service is not None:
            resolved = await self._resolve_gift_monochrome(
                gift_attrs_service, gift_name, number, results,
            )
            if resolved is not None:
                return json.dumps(resolved, ensure_ascii=False)

            # Number given but could not be resolved (404 / not a gift
            # page). Still return the collection list so the model can
            # answer with the types that exist and ask for a valid link.
            return json.dumps(
                {
                    "gift_name": gift_name,
                    "number": number,
                    "resolved": False,
                    "note": (
                        "не удалось определить модель и фон подарка по "
                        "этому номеру. покажи какие типы монохрома есть у "
                        "коллекции и попроси точную ссылку на подарок"
                    ),
                    "count": len(results),
                    "results": results,
                },
                ensure_ascii=False,
            )

        return json.dumps(
            {
                "gift_name": gift_name,
                "count": len(results),
                "results": results,
            },
            ensure_ascii=False,
        )

    @staticmethod
    def _coerce_int(value: Any) -> int | None:
        """Coerce an LLM-provided number into a positive int, else None.

        Models occasionally pass ``"3387"`` (string) or ``3387.0`` (float)
        despite the schema; normalize defensively.
        """
        try:
            n = int(value)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return None
        return n if n > 0 else None

    async def _resolve_gift_monochrome(
        self,
        gift_attrs_service: GiftAttrsService,
        gift_name: str,
        number: int,
        collection_results: list[dict[str, Any]],
    ) -> dict[str, Any] | None:
        """Resolve a specific gift number to its concrete monochrome type.

        Fetches the t.me preview page for ``<gift_name>-<number>``, reads
        its Model / Backdrop / Symbol, then filters the collection matrix
        to the matching row.

        Args:
            collection_results: The full collection monochrome matrix
                (already fetched with the canonical name). Filtered
                locally to avoid an extra API round-trip.

        Returns:
            A result dict with ``resolved: True`` and the single matching
            ``type``, or ``None`` if the gift page could not be resolved
            (404 / unknown number).
        """
        from core.exceptions import GiftAttrsError

        try:
            attrs = await gift_attrs_service.get_attributes(gift_name, number)
        except GiftAttrsError as e:
            logger.warning(
                "LLMService: gift attrs fetch failed for %s-%d: %s",
                gift_name, number, e,
            )
            return None
        if attrs is None:
            return None

        model = attrs.get("model")
        backdrop = attrs.get("backdrop")
        matched = _filter_monochrome(
            collection_results, model=model, backdrop=backdrop,
        )

        payload: dict[str, Any] = {
            "gift_name": gift_name,
            "number": number,
            "resolved": True,
            "model": model,
            "backdrop": backdrop,
        }
        if attrs.get("symbol"):
            payload["symbol"] = attrs["symbol"]

        if matched:
            payload["type"] = matched[0].get("type")
            payload["results"] = matched
        else:
            # Model + backdrop are known but GiftWiki has no monochrome
            # entry for this exact combination — say so honestly rather
            # than falling back to the generic list.
            payload["type"] = None
            payload["note"] = (
                "GiftWiki не классифицирует эту модель+фон как монохром. "
                "Скорее всего это НЕ монохром."
            )
        return payload

    async def close(self) -> None:
        """Close the shared HTTP session."""
        if self._session and not self._session.closed:
            await self._session.close()


def _service_supports(fn_name: str, services: dict[str, Any]) -> bool:
    """Return True if the service backing ``fn_name`` is available."""
    if fn_name in ("get_floor_prices", "get_collection_floors"):
        return services.get("price_service") is not None
    if fn_name == "convert_currency":
        return services.get("crypto_service") is not None
    if fn_name == "get_currency_history":
        return services.get("crypto_service") is not None
    if fn_name == "get_monochrome":
        return services.get("giftwiki_service") is not None
    if fn_name in ("get_market_snapshot", "get_price_history"):
        return services.get("moomin_service") is not None
    return False


def _normalize_query(query: str) -> str:
    """Lowercase + strip. Used for case-insensitive substring match."""
    return query.strip().lower()


# Full Cyrillic→Latin transliteration table for matching Russian slang
# against English collection names ("пепе" → "pepe", "лягушка" → "lyagushka").
_RU_EN_TABLE: dict[str, str] = {
    "а": "a", "б": "b", "в": "v", "г": "g", "д": "d", "е": "e", "ё": "yo",
    "ж": "zh", "з": "z", "и": "i", "й": "y", "к": "k", "л": "l", "м": "m",
    "н": "n", "о": "o", "п": "p", "р": "r", "с": "s", "т": "t", "у": "u",
    "ф": "f", "х": "x", "ц": "ts", "ч": "ch", "ш": "sh", "щ": "sch",
    "ъ": "", "ы": "y", "ь": "", "э": "e", "ю": "yu", "я": "ya",
}


def _translit(s: str) -> str:
    """Full Cyrillic→Latin transliteration for matching collection names."""
    return "".join(_RU_EN_TABLE.get(c, c) for c in s.lower())


# Fuzzy-match thresholds. ``_FUZZY_TOKEN_RATIO`` covers single-word typo
# queries ("lolpop" → "lollipop", ratio ≈ 0.94). ``_FUZZY_FULL_RATIO`` is
# for when the user types the whole collection name with a small error.
_FUZZY_TOKEN_RATIO = 0.82
_FUZZY_FULL_RATIO = 0.85

# Score tiers — substring/translit matches are exact (1.0), fuzzy matches
# are scored by ratio, non-matches are 0.
_SCORE_EXACT = 1.0


def _fuzzy_score(query_norm: str, name: str) -> float:
    """Return a relevance score (0.0–1.0) for ``query_norm`` vs ``name``.

    Higher score = closer match. Order of checks, cheapest first:

    1. Exact substring on the original name ("snake" → "Lunar Snake").
    2. Substring after transliterating the query ("пепе" → "pepe" →
       "Plush Pepe").
    3. Substring of the transliterated name against the query.
    4. Token-level fuzzy ratio: best ``SequenceMatcher.ratio()`` between
       the query and any whitespace-separated token of the name.
    5. Whole-name fuzzy ratio.

    Returns 0.0 if no check passes the minimum threshold.
    """
    if not query_norm or not name:
        return 0.0

    # Substring matches — best possible score.
    if query_norm in name:
        return _SCORE_EXACT
    q_translit = _translit(query_norm)
    if q_translit and q_translit in name:
        return _SCORE_EXACT
    if _translit(name).find(query_norm) >= 0:
        return _SCORE_EXACT

    # Token-level fuzzy: compare the query against each word of the name.
    best_token = 0.0
    for token in name.split():
        if not token:
            continue
        ratio = SequenceMatcher(None, query_norm, token).ratio()
        if ratio > best_token:
            best_token = ratio

    # Whole-name fuzzy for full-name queries with a typo.
    full_ratio = SequenceMatcher(None, query_norm, name).ratio()

    # Return the best score above the respective thresholds, or 0.
    if best_token >= _FUZZY_TOKEN_RATIO:
        return best_token
    if full_ratio >= _FUZZY_FULL_RATIO:
        return full_ratio
    return 0.0


def _fuzzy_match(query_norm: str, name: str) -> bool:
    """Return True if ``query_norm`` matches collection ``name``."""
    return _fuzzy_score(query_norm, name) > 0.0


def _filter_and_pick(prices: list, query_norm: str | None, cap: int) -> list:
    """Filter prices by query and sort by relevance.

    When ``query_norm`` is given, only collections whose name matches the
    query (substring, transliteration, or fuzzy typo match) are returned
    — and ALL matches, ignoring ``cap``. Results are sorted by descending
    relevance score so the LLM sees the best matches first (e.g. "lolpop"
    → "LolPop" before "Hypno Lollipop").
    """
    if not query_norm:
        return prices[:cap]

    scored: list[tuple[float, Any]] = []
    for p in prices:
        name = (p.gift_name or "").lower()
        score = _fuzzy_score(query_norm, name)
        if score > 0.0:
            scored.append((score, p))

    # Sort by score descending — best matches first.
    scored.sort(key=lambda x: x[0], reverse=True)
    return [p for _, p in scored]


def _filter_monochrome(
    rows: list[dict[str, Any]],
    *,
    model: str | None,
    backdrop: str | None,
) -> list[dict[str, Any]]:
    """Match monochrome matrix rows to a specific model + backdrop.

    Comparison is case-insensitive. If a field is missing from the gift
    page, that constraint is skipped (so a model-only match still works).
    """
    if not rows:
        return []
    m = model.strip().lower() if model else None
    b = backdrop.strip().lower() if backdrop else None

    def _eq(row_val: Any, want: str | None) -> bool:
        if want is None:
            return True
        rv = (row_val or "").strip().lower()
        return rv == want

    return [
        r for r in rows
        if _eq(r.get("model_name"), m) and _eq(r.get("backdrop_name"), b)
    ]


# ── Price-history aggregation helpers ─────────────────────────────────
# The Moomin candles endpoint can return hundreds of bars (e.g. 1h over a
# year = ~8760). Rather than dumping them raw (token explosion), we
# summarize into a compact trend overview and downsample the close-price
# series to a fixed number of points the model can describe.

_DEFAULT_HISTORY_DAYS = 7
_DEFAULT_MAX_DAYS = 1095
_HISTORY_SERIES_POINTS = 12
# Direction threshold: |change_pct| below this is "flat".
_FLAT_THRESHOLD_PCT = 2.0

# Per-interval max lookback (mirrors services/moomin_service.py).
_INTERVAL_MAX_DAYS: dict[str, int] = {"5m": 31, "1h": 366, "1d": 1095}


def _to_float(value: Any) -> float | None:
    """Parse a candle OHLC value (string or number) into a float."""
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _summarize_candles(
    bars: list[dict[str, Any]],
) -> dict[str, Any]:
    """Compute a compact trend summary over OHLC bars.

    Returns ``open/high/low/close`` (first open, last close, period
    extrema), ``change_pct`` (close vs open), and ``direction``
    (up/down/flat with a ±2% threshold). Empty input → ``None`` fields.
    """
    if not bars:
        return {
            "open": None, "high": None, "low": None, "close": None,
            "change_pct": None, "direction": None,
        }
    opens = [_to_float(b.get("open")) for b in bars]
    highs = [_to_float(b.get("high")) for b in bars]
    lows = [_to_float(b.get("low")) for b in bars]
    closes = [_to_float(b.get("close")) for b in bars]

    open_val = opens[0]
    close_val = closes[-1] if closes else None
    high_val = max((h for h in highs if h is not None), default=None)
    low_val = min((l for l in lows if l is not None), default=None)

    change_pct: float | None = None
    direction = "flat"
    if open_val and close_val:
        change_pct = round((close_val - open_val) / open_val * 100, 2)
        if change_pct > _FLAT_THRESHOLD_PCT:
            direction = "up"
        elif change_pct < -_FLAT_THRESHOLD_PCT:
            direction = "down"

    def _fmt(v: float | None) -> float | None:
        return round(v, 4) if v is not None else None

    return {
        "open": _fmt(open_val),
        "high": _fmt(high_val),
        "low": _fmt(low_val),
        "close": _fmt(close_val),
        "change_pct": change_pct,
        "direction": direction,
    }


def _downsample_close_series(
    bars: list[dict[str, Any]], points: int,
) -> list[dict[str, Any]]:
    """Return ``points`` evenly-spaced ``{t, close}`` samples from bars.

    Picks roughly evenly-spaced bars (start, ..., end) so the model sees
    the shape of the trend without hundreds of rows. Each sample keeps
    the bar's ``start`` timestamp and ``close`` price. Fewer bars than
    ``points`` → returned as-is.
    """
    if not bars:
        return []
    n = len(bars)
    if n <= points:
        out = []
        for b in bars:
            close = _to_float(b.get("close"))
            if close is None:
                continue
            out.append({"t": b.get("start"), "close": round(close, 4)})
        return out

    out: list[dict[str, Any]] = []
    # Evenly spaced indices 0..n-1, ``points`` total, always including
    # first and last.
    for i in range(points):
        idx = round(i * (n - 1) / (points - 1))
        b = bars[idx]
        close = _to_float(b.get("close"))
        if close is None:
            continue
        out.append({"t": b.get("start"), "close": round(close, 4)})
    return out
