#!/usr/bin/env python3
"""Interactive login — produces a Kurigram string session file.

Run once to bootstrap the user session:

    python scripts/login.py

Prompts for api_id / api_hash (or reads from .env) and the phone number,
performs the Telegram login flow (code + optional 2FA), exports the session
string and writes it to ``user/session/session.string``.

After this, the bot reads that file automatically — no SESSION_STRING in
.env is needed.
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

# Allow running from project root without install
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from user.session.store import SessionStore  # noqa: E402


async def _do_login(api_id: int, api_hash: str, store: SessionStore) -> None:
    try:
        from kurigram import Client
    except ImportError:
        try:
            from pyrogram import Client  # type: ignore
        except ImportError:
            print("ERROR: install kurigram first: pip install kurigram")
            sys.exit(1)

    # Use a throwaway in-memory client to perform the interactive login.
    client = Client(
        name="oclp_login",
        api_id=api_id,
        api_hash=api_hash,
        in_memory=True,
    )
    await client.start()
    session_string = await client.export_session_string()
    await client.stop()

    store.save(session_string)
    print(f"\n✓ Session saved to {store.path}")
    print("  You can now start the bot — it will pick up this file automatically.")


def main() -> None:
    store = SessionStore()

    if store.exists():
        print(f"Existing session found at {store.path}")
        answer = input("Re-login and overwrite? [y/N] ").strip().lower()
        if answer != "y":
            print("Aborted.")
            return

    api_id_str = os.environ.get("API_ID") or input("API ID: ").strip()
    api_hash = os.environ.get("API_HASH") or input("API HASH: ").strip()

    if not api_id_str or not api_hash:
        print("ERROR: API_ID and API_HASH are required.")
        sys.exit(1)

    try:
        api_id = int(api_id_str)
    except ValueError:
        print("ERROR: API_ID must be an integer.")
        sys.exit(1)

    print("\nStarting Telegram login flow — you'll be asked for phone/code/2FA.\n")
    asyncio.run(_do_login(api_id, api_hash, store))


if __name__ == "__main__":
    main()
