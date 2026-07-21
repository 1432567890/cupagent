from __future__ import annotations


class cupagentError(Exception):
    """Base exception for the application."""


class MarketAuthError(cupagentError):
    """Failed to authenticate with a marketplace API."""


class MarketApiError(cupagentError):
    """Marketplace API returned an error or unexpected response."""


class CacheError(cupagentError):
    """Redis cache operation failed."""


class GiftAttrsError(cupagentError):
    """Failed to fetch or parse a Telegram collectible gift page."""


class MoominError(cupagentError):
    """Moomin Market API error."""


class DatabaseError(cupagentError):
    """Database operation failed."""


class SessionExpiredError(cupagentError):
    """Telegram initData has expired and needs refresh."""


class ConfigError(cupagentError):
    """Missing or invalid configuration."""


class LLMError(cupagentError):
    """LLM provider returned an error or unexpected response."""


class ToolCallError(LLMError):
    """LLM requested a tool call but the required service is unavailable."""
