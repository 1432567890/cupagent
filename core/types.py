from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum


class MarketName(str, Enum):
    """Marketplace identifiers. Inherits from str so comparisons like
    ``market.value`` and ``market == "grapes"`` both work naturally."""

    GRAPES = "grapes"
    MRKT = "mrkt"
    PORTAL = "portal"
    GETGEMS = "getgems"
    TONNEL = "tonnel"
    XGIFT = "xgift"

    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True)
class FloorPrice:
    """Floor price entry for a single gift collection."""

    gift_name: str
    market: MarketName
    price: float | None  # TON, None = not listed
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
