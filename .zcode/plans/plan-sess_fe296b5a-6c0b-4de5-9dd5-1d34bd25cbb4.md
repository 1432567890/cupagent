# План: интеграция Moomin Market API + переработка промпта

## Контекст

Архив `moomin-market-api.7z` — это skill-бандл для агрегатора рыночных данных `api.moomin.cfd/market/v1`. API проверен: `healthz`, `collections`, `snapshot` (кросс-маркет цены + `direct_floor`), `candles` (OHLC), `observations` — работают с ключом из архива. Quote asset — TON (= GRAM 1:1). Slugs — lowercase без пробелов (`Artisan Brick` → `artisanbrick`).

Интегрирую по образцу `GiftWikiService` (async, `aiohttp`, `X-API-Key`, Redis-кеш, lazy session, tool в `llm_service.py`, keyword-trigger skill). Параллельно — полная переработка `skills/instruction.md` для естественности.

---

## Часть A — Moomin Market API

### A1. `core/constants.py`
Добавить:
```python
MOOMIN_BASE_URL = "https://api.moomin.cfd/market/v1"
REDIS_MOOMIN_KEY_PREFIX = "cupagent:moomin"
MOOMIN_COLLECTIONS_TTL = 3600    # 1h — список коллекций меняется редко
MOOMIN_SNAPSHOT_TTL = 60         # 1m — цены волатильны
MOOMIN_CANDLES_TTL = 300         # 5m — исторические свечи
MOOMIN_HTTP_TIMEOUT = 20         # seconds
```

### A2. `core/exceptions.py`
```python
class MoominError(cupagentError):
    """Moomin Market API error."""
```

### A3. `config/settings.py` + `.env` + `.env.example`
Добавить `MOOMIN_API_KEY: str = ""`. В `.env` вписать ключ из архива (`market_9b8148b0b906615900028c0d73c4b67e1722c2409b3637faddd9c501011f342d`). В `.env.example` — пустой с комментарием.

### A4. `services/moomin_service.py` (новый, ~250 строк)
По образцу `giftwiki_service.py`:
- `MoominService.__init__(*, api_key, redis, base_url)`
- Ленивый `aiohttp.ClientSession` с заголовком `X-API-Key`
- `_get(path, params)` — низкоуровневый GET с обработкой 401/403/404/429/503/5xx → `MoominError`
- `_slug(name)` — lowercase + удалить пробелы (`Artisan Brick` → `artisanbrick`)
- `_cached(key, ttl, producer)` — общий Redis-кеш (как в GiftWiki)
- Публичные методы:
  - `get_collections(limit=250)` → список `[{slug, title, updated_at}]`
  - `get_snapshot(collection, include_derived=False)` → кросс-маркет цены + `direct_floor`; проекция: только `market`, `price_ton`, `observed_at` (nanoTON выбрасываем — LLM не нужен)
  - `get_candles(collection, market, interval, from_dt, to_dt)` → OHLC
  - `close()`

### A5. `services/llm_service.py`
**Два новых tool-схема** (компактно, как существующие):

1. `get_market_snapshot` — параметры: `collection` (имя/слаг), опционально `include_derived`. Описание: «актуальные кросс-маркет цены коллекции + прямой floor; используй для вопроса "цена сейчас по всем маркетам"».
2. `get_price_history` — параметры: `collection`, `market` (grapes/mrkt/portals/getgems/tonnel/xgift), `interval` (5m/1h/1d), `days` (число дней, по умолчанию 7, на 1d — до 366). Описание: «OHLC-свечи за период; используй для тренд/динамика/"как менялась цена"/"росла ли"/"упала ли"».

**Маршрутизация маркетов**: мапа `grapes/mrkt/portals/getgems/tonnel/xgift` → slug маркета в Moomin API (`portal` → `portals`). 
**Реализация**: `_tool_market_snapshot`, `_tool_price_history` — компактная JSON-проекция:
- snapshot: `{collection, quote_asset, prices: [{market, price, observed_at}...], direct_floor: {market, price}}` (маркеты с пустой ценой пропускаются — как в `_tool_floor_prices`)
- history: `{collection, market, interval, bars, period, summary: {open, high, low, close, change_pct, direction}, series: [12 прореженных точек close]}` — не сырой дамп свечей, а готовая сводка + ряд для описания формы тренда. Процент изменения `(close-open)/open*100`, направление up/down/flat по порогу ±2%.

**Регистрация**:
- добавить схемы в `_build_tools` (когда `moomin_service` есть)
- добавить имена в `_KNOWN_TOOL_NAMES`
- добавить `_service_supports` для новых имён → `moomin_service`
- добавить `moomin_service` в сигнатуры `chat()` и `_execute_tool_call()`

### A6. `bot/handlers/_chat_core.py`
Добавить параметр `moomin_service=None` в `generate_reply()` и прокинуть в `llm.chat()`.

### A7. `bot/handlers/chat.py` + `bot/handlers/guest_chat.py`
Добавить `moomin_service=None` в сигнатуры и в вызовы `generate_reply`/`llm.chat` (по образцу `giftwiki_service`).

### A8. `main.py`
- `_init_moomin_service(settings, redis_client)` — возвращает `None` если нет ключа (как `_init_giftwiki_service`)
- Создать и `dp["moomin_service"] = moomin_service`
- `moomin_service.close()` в shutdown и `finally`
- В startup-лог добавить `moomin=%s`

### A9. `skills/market-history.md` (новый)
Skill-документ под нашу архитектуру (НЕ сырой api.md из архива — тот для curl-style агента). Объясняет LLM: когда запрос про тренд/динамику/«как менялась цена» → `get_price_history`; когда «цена сейчас» — можно `get_market_snapshot` ИЛИ существующий `get_floor_prices`. Trigger-ключевые слова: тренд, динамика, история цен, росла, упала, изменилась, grew, fell, trend, history, как менялась, за неделю, за месяц.

### A10. Регистрация skill в `llm_service.py`
Добавить `"market-history"` в `_SKILL_TRIGGERS` с триггерами (трененд/динамика/история цен/росла/упала/как менялась/за неделю/за месяц/grew/fell/trend/history).

---

## Часть B — Полная переработка `skills/instruction.md`

Принцип: **промпт вокруг голоса и принципов, а не каталог запретов**. Guardrails (безопасность, обязательные tools, язык, запрет выдумок, лояльность Grapes, запрет редиректа в GiftWiki) сохраняются полностью — но позитивной формулировкой и без 10-кратного повторения. Что убираю/ужимаю:

- **Каталог ~25 стоп-фраз** («че по делу», «братан», «чем могу помочь»...) → одно правило: «общайся как живой человек в чате, без колл-центр-штампов и панибратства». Постобработка в `_chat_core.py` (`_strip_giftwiki_redirects`, `wrap_gift_links`, `_wrap_prices_in_code`) уже делает работу в коде — не нужно дублировать в промпте.
- **Шаблон-пример вывода** (Swiss Watch 33.77...) — убираю, LLM копирует дословно.
- **Дублирующиеся блоки правил** (правила форматирования повторяются 3 раза) → одна компактная секция.
- **10 повторений «НИКОГДА не выдумывай»** → одно правило «без выдумок: нет данных — так и скажи».
- **Длинные инструкции по сленгу в теле промпта** → переносятся в lazy-skills (slang/gift-numbers уже подключаются по триггерам).

**Сохраняю полностью**: роль Grapes, правила безопасности (высший приоритет), обязательный вызов tools для цен/монохрома/курса, правило языка, запрет выдумок, запрет редиректа в GiftWiki (один чёткий абзац вместо 3), лояльность Grapes нейтральная, форматирование валют (тикер GRAM/TON в верхнем регистре, фиат словами), правило HTML.

**Голос**: компетентный, ровный, короткий, без колл-центр-тони — как живой человек в Telegram, который разбирается в теме. 1–3 предложения для простого ответа. Не здороваться и не прощаться в каждом сообщении. Варьировать структуру ответов.

**Новый Moomin блок** добавляю в секцию про инструменты: «для тренд/динамика цены — `get_price_history`; для актуальной кросс-маркет цены — `get_market_snapshot` или `get_floor_prices`».

Итоговый объём: ~150 строк против ~390 сейчас — фокус на принципах и голосе, guardrails сжаты без потери смысла.

---

## Что НЕ трогаю
- Существующие markets/* клиенты и PriceService — Moomin отдельный источник (агрегатор), не пересекается.
- Постобработку вывода в `_chat_core.py` — она уже покрывает большую часть «запрещённых» правил кодом.
- Остальные skills (slang, gift-numbers, orig-accounts, grapes-features, market-prices) — не трогаю.

## Файлы (итог)
- **Новые**: `services/moomin_service.py`, `skills/market-history.md`
- **Изменены**: `core/constants.py`, `core/exceptions.py`, `config/settings.py`, `.env`, `.env.example`, `services/llm_service.py`, `bot/handlers/_chat_core.py`, `bot/handlers/chat.py`, `bot/handlers/guest_chat.py`, `main.py`, `skills/instruction.md`

## Проверка
- Импорт всего и синтаксис: `python -c "import main"` без запуска бота
- Юнит-проверка проекций Moomin на реальном ответе API (snapshot artisanbrick + candles) — без отправки в Telegram