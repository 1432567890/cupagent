"""SQLAlchemy models for floor price storage."""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import Float, String, DateTime, func
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class FloorPriceModel(Base):
    """Persisted floor price for a gift collection on a specific market."""

    __tablename__ = "floor_prices"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    gift_name: Mapped[str] = mapped_column(String(255), index=True)
    market: Mapped[str] = mapped_column(String(50), index=True)
    price: Mapped[float | None] = mapped_column(Float, nullable=True)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), onupdate=func.now()
    )
