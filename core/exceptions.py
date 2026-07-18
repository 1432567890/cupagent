from __future__ import annotations


class CupagentError(Exception):
    """Base exception for the application."""


class MarketAuthError(CupagentError):
    """Failed to authenticate with a marketplace API."""


class MarketApiError(CupagentError):
    """Marketplace API returned an error or unexpected response."""


class CacheError(CupagentError):
    """Redis cache operation failed."""


class GiftAttrsError(CupagentError):
    """Failed to fetch or parse a Telegram collectible gift page."""


class DatabaseError(CupagentError):
    """Database operation failed."""


class SessionExpiredError(CupagentError):
    """Telegram initData has expired and needs refresh."""


class ConfigError(CupagentError):
    """Missing or invalid configuration."""


class LLMError(CupagentError):
    """LLM provider returned an error or unexpected response."""


class ToolCallError(LLMError):
    """LLM requested a tool call but the required service is unavailable."""
