from __future__ import annotations


class oclpError(Exception):
    """Base exception for the application."""


class MarketAuthError(oclpError):
    """Failed to authenticate with a marketplace API."""


class MarketApiError(oclpError):
    """Marketplace API returned an error or unexpected response."""


class CacheError(oclpError):
    """Redis cache operation failed."""


class GiftAttrsError(oclpError):
    """Failed to fetch or parse a Telegram collectible gift page."""


class DatabaseError(oclpError):
    """Database operation failed."""


class SessionExpiredError(oclpError):
    """Telegram initData has expired and needs refresh."""


class ConfigError(oclpError):
    """Missing or invalid configuration."""


class LLMError(oclpError):
    """LLM provider returned an error or unexpected response."""


class ToolCallError(LLMError):
    """LLM requested a tool call but the required service is unavailable."""
