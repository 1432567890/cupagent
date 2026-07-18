"""Persistent store for the Kurigram string session.

The session string is a single base64-ish line that encodes the
authorization key + DC + user id. We persist it to a plain text file at
``user/session/session.string`` so that:

  * the bot doesn't need SESSION_STRING in .env on every restart,
  * a freshly-logged-in session (via ``scripts/login.py``) is reused
    automatically by the running process,
  * re-runs of ``login.py`` overwrite it transparently when re-authenticating.

This module is sync (file I/O only, no network) — safe to call from startup.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

# Default location relative to the project root.
_DEFAULT_PATH = Path(__file__).resolve().parent / "session.string"


class SessionStore:
    """Load / save the Kurigram string session from disk."""

    def __init__(self, path: str | Path | None = None) -> None:
        self._path = Path(path) if path else _DEFAULT_PATH

    @property
    def path(self) -> Path:
        """Absolute path to the session file."""
        return self._path

    def exists(self) -> bool:
        """True if a session file is already present on disk."""
        return self._path.is_file() and self._path.stat().st_size > 0

    def load(self) -> str | None:
        """Read the session string from disk.

        Returns:
            The session string, or ``None`` if the file is missing/empty.
        """
        if not self.exists():
            return None
        try:
            value = self._path.read_text(encoding="utf-8").strip()
            return value or None
        except OSError:
            logger.exception("SessionStore: failed to read %s", self._path)
            return None

    def save(self, session_string: str) -> None:
        """Persist the session string atomically.

        Writes to a temp file and renames, so a crash mid-write can't
        corrupt the existing session. Permissions are restricted to the
        owner (mode 0600) since this grants account access.

        Args:
            session_string: The Kurigram/Pyrogram session string to store.
        """
        if not session_string:
            logger.warning("SessionStore: refusing to save empty session string")
            return

        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._path.with_suffix(".tmp")

        # Write atomically: temp file → rename
        tmp.write_text(session_string, encoding="utf-8")
        os.chmod(tmp, 0o600)
        os.replace(tmp, self._path)
        logger.info("SessionStore: saved session to %s", self._path)

    def clear(self) -> None:
        """Delete the session file (forces a fresh login on next start)."""
        try:
            self._path.unlink(missing_ok=True)
            logger.info("SessionStore: removed %s", self._path)
        except OSError:
            logger.exception("SessionStore: failed to remove %s", self._path)
