from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any

import aiohttp

from core.types import FloorPrice, MarketName

logger = logging.getLogger(__name__)


class BaseMarketClient(ABC):
    """Abstract base class for all marketplace API clients."""

    market_name: MarketName
    base_url: str

    def __init__(self, token: str) -> None:
        self._token = token
        self._session: aiohttp.ClientSession | None = None

    @property
    def session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            # NOTE: do NOT pass base_url here — _request() already builds
            # absolute URLs from self.base_url, and aiohttp refuses to merge
            # an absolute URL onto a session that has base_url set.
            self._session = aiohttp.ClientSession(
                headers=self._default_headers,
                timeout=aiohttp.ClientTimeout(total=30),
            )
        return self._session

    @property
    def _default_headers(self) -> dict[str, str]:
        return {
            "Accept": "application/json",
            "Content-Type": "application/json",
        }

    @abstractmethod
    def _auth_headers(self) -> dict[str, str]:
        """Return headers needed for authentication with this market."""

    @abstractmethod
    async def fetch_floor_prices(self) -> list[FloorPrice]:
        """Fetch all gift collection floor prices from this market."""

    @abstractmethod
    async def authenticate(self) -> None:
        """Perform authentication / session creation if needed."""

    async def _request(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        json_body: dict[str, Any] | None = None,
        retry_on_401: bool = True,
    ) -> Any:
        """Make an authenticated HTTP request to the market API.

        On HTTP 401/403 (when ``retry_on_401`` is True), automatically
        re-authenticates via :meth:`authenticate` and retries the request
        exactly once. This matches how amrkt handles MRKT token expiry:
        the session token can be invalidated server-side at any time, so
        a single self-healing retry keeps the client resilient without
        surfacing ``MarketAuthError`` to the caller.
        """
        url = f"{self.base_url}{path}"
        headers = {**self._default_headers, **self._auth_headers()}

        logger.debug(
            "Market request: %s %s (market=%s)", method, path, self.market_name
        )

        async with self.session.request(
            method, url, headers=headers, params=params, json=json_body
        ) as resp:
            if resp.status in (401, 403):
                if retry_on_401:
                    logger.warning(
                        "%s: HTTP %d on %s — re-authenticating and retrying once",
                        self.market_name, resp.status, path,
                    )
                    # Drop any cached initData/token first so authenticate()
                    # fetches fresh credentials instead of reusing the ones
                    # that just got rejected.
                    invalidate = getattr(self, "_invalidate_auth", None)
                    if invalidate is not None:
                        await invalidate()
                    await self.authenticate()
                    return await self._request(
                        method, path,
                        params=params, json_body=json_body,
                        retry_on_401=False,
                    )
                from core.exceptions import MarketAuthError

                raise MarketAuthError(
                    f"Auth failed for {self.market_name}: HTTP {resp.status}"
                )
            resp.raise_for_status()
            if resp.content_type == "application/json":
                return await resp.json()
            return await resp.text()

    async def close(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()

    async def __aenter__(self) -> "BaseMarketClient":
        return self

    async def __aexit__(self, *args: Any) -> None:
        await self.close()
