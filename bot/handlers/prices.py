"""Price commands handler."""

from __future__ import annotations

from aiogram import Router, F
from aiogram.types import Message

from core.types import MarketName
from services.price_service import PriceService

router = Router()

# Human-readable display names for markets.
_MARKET_DISPLAY: dict[MarketName, str] = {
    MarketName.GRAPES: "Grapes",
    MarketName.MRKT: "MRKT",
    MarketName.PORTAL: "Portals",
    MarketName.GETGEMS: "GetGems",
    MarketName.TONNEL: "Tonnel",
    MarketName.XGIFT: "XGift",
}


def _market_from_text(text: str) -> MarketName | None:
    """Parse market name from command text."""
    lower = text.lower().strip()
    parts = lower.split()
    if len(parts) < 2:
        return None
    arg = parts[1].lower()
    mapping = {
        "grapes": MarketName.GRAPES,
        "mrkt": MarketName.MRKT,
        "portal": MarketName.PORTAL,
        "getgems": MarketName.GETGEMS,
        "tonnel": MarketName.TONNEL,
        "xgift": MarketName.XGIFT,
        "all": None,
    }
    return mapping.get(arg)


@router.message(F.text.lower().startswith("/prices"))
async def cmd_prices(message: Message, price_service: PriceService | None = None) -> None:
    """Handle /prices command — show floor prices from markets.

    ``price_service`` is injected from the dispatcher's workflow_data
    (set in main.py via ``dp["price_service"] = ...``) — aiogram resolves
    it by parameter name. May be ``None`` when no Kurigram session is
    configured (then MRKT/Portal are unavailable).
    """
    if price_service is None:
        await message.answer("⚠️ сервис цен не инициализирован")
        return

    text = message.text or ""
    market = _market_from_text(text)

    if market is not None:
        prices = await price_service.get_floor_prices(market)
        display_name = _MARKET_DISPLAY[market]

        if not prices:
            await message.answer(
                f"📊 нет данных по {display_name} "
                f"(обновляется...)"
            )
            return

        lines = [f"<b>📊 {display_name} — Floor Prices</b>\n"]
        for p in prices[:50]:  # limit to 50 to avoid message too long
            price_str = f"{p.price:.2f} GRAM" if p.price is not None else "—"
            lines.append(f"{p.gift_name}: <code>{price_str}</code>")
        if len(prices) > 50:
            lines.append(f"\n  ...и ещё {len(prices) - 50} коллекций")

        await message.answer("\n".join(lines))
    else:
        # Show summary across all markets
        summary_lines = ["<b>📊 Floor Prices Summary</b>\n"]
        for m in MarketName:
            prices = await price_service.get_floor_prices(m)
            count = len([p for p in prices if p.price is not None])
            display_name = _MARKET_DISPLAY[m]
            summary_lines.append(
                f"  <b>{display_name}</b>: {count} коллекций с ценами"
            )
        summary_lines.append(
            "\nиспользуй /prices grapes, /prices mrkt, /prices portal, "
            "/prices tonnel или /prices xgift для детальных данных"
        )
        await message.answer("\n".join(summary_lines))
