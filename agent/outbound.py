"""Outbound calls: the phone agent calls someone for its owner and holds the conversation.

    await place_call(to="+15555550100", reason="You're running about 10 minutes late to coffee.")

Uses the same Vapi assistant and number as inbound, with the live situation (situation.py)
plus the reason for the call in {notes}, and an opener that gets to the point.

Guardrails:
- OUTBOUND_CALLS_ENABLED=1 in .env, or nothing is dialed (kill switch; off by default)
- only numbers in DEMO_CONTACTS_JSON can be called
- one call per (number, dedupe_key) — e.g. one "running late" call per event — tracked in
  out/outbound_calls.json
"""

import asyncio
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import httpx

import situation

logger = logging.getLogger("outbound")

HERE = Path(__file__).parent
LEDGER = HERE / "out" / "outbound_calls.json"
VAPI_URL = "https://api.vapi.ai/call"
from dotenv import load_dotenv as _load_dotenv

_load_dotenv(Path(__file__).resolve().parent / ".env")  # SPEAKER_NAME etc. must be set before use, whoever imports us
SPEAKER = os.getenv("SPEAKER_NAME", "Michael")


class CallRefused(Exception):
    """A guardrail stopped the call; the message says which."""


def _ledger() -> dict:
    try:
        return json.loads(LEDGER.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def _record(key: str, entry: dict) -> None:
    data = _ledger()
    data[key] = entry
    LEDGER.parent.mkdir(exist_ok=True)
    LEDGER.write_text(json.dumps(data, indent=2))


def build_call(to: str, reason: str, name: str | None, notes: str, opener: str | None) -> dict:
    who = name or "the person you're calling"
    first = opener or (f"Hey {name.split()[0]}, it's {SPEAKER}." if name else f"Hey, it's {SPEAKER}.")
    return {
        "assistantId": os.environ["VAPI_ASSISTANT_ID"],
        "phoneNumberId": os.getenv("VAPI_OUTBOUND_PHONE_NUMBER_ID") or os.environ["VAPI_PHONE_NUMBER_ID"],
        "customer": {"number": to, **({"name": name} if name else {})},
        "assistantOverrides": {
            "firstMessage": first,
            "firstMessageMode": "assistant-speaks-first",
            "variableValues": {
                "speaker_name": SPEAKER,
                "caller_name": who,
                "notes": f"You placed this call to {who}. Why you're calling: {reason}\n"
                         f"Say why you're calling right after the greeting, then talk it through.\n{notes}",
            },
            "metadata": {"direction": "outbound", "caller_name": name, "reason": reason},
        },
    }


async def place_call(to: str, reason: str, *, dedupe_key: str | None = None, opener: str | None = None) -> dict:
    if os.getenv("OUTBOUND_CALLS_ENABLED") != "1":
        raise CallRefused("outbound calls are off (set OUTBOUND_CALLS_ENABLED=1 in .env)")
    name = situation.local_contact(to)
    if name is None:
        raise CallRefused(f"{to} is not in DEMO_CONTACTS_JSON; only listed contacts can be called")
    key = f"{situation.digits(to)}:{dedupe_key or reason}"
    if key in _ledger():
        raise CallRefused(f"already called {name} for this ({_ledger()[key]['at']}); not calling twice")
    ctx = await situation.situation_for_caller(to)
    payload = build_call(to, reason, name, ctx["notes"], opener)
    async with httpx.AsyncClient(timeout=20) as client:
        r = await client.post(VAPI_URL, json=payload,
                              headers={"Authorization": f"Bearer {os.environ['VAPI_API_KEY']}"})
    if r.is_error:
        raise RuntimeError(f"Vapi refused the call: {r.status_code} {r.text}")
    call = r.json()
    _record(key, {"at": datetime.now(timezone.utc).isoformat(), "to": to, "name": name,
                  "reason": reason, "call_id": call.get("id"), "status": call.get("status")})
    logger.info("outbound call to %s (%s): %s", name, to, call.get("id"))
    return {"call_id": call.get("id"), "status": call.get("status"), "to": to, "name": name}


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv(HERE / ".env")
    logging.basicConfig(level=logging.INFO)
    if len(sys.argv) < 3:
        sys.exit('usage: uv run outbound.py +1XXXXXXXXXX "why you are calling"')
    print(asyncio.run(place_call(sys.argv[1], sys.argv[2])))
