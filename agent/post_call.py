"""After a call: hand the transcript to UFO, which follows up using the river_* tools.

    await hand_off_call(report)   # report = Vapi "end-of-call-report" message dict

UFO runs locally (ufoctl serve on :8710). We POST the task to its surface and return the turn id
right away; UFO works in the background (look up context, save a call note to GBrain, draft
follow-up emails into Gmail Drafts; it never sends).

    uv run post_call.py [call_id]    # print that call's latest UFO result
"""

import asyncio
import logging
import re
import os
import sys
import uuid
from pathlib import Path

import httpx

logger = logging.getLogger("post_call")

UFO_URL = os.getenv("UFO_URL", "http://127.0.0.1:8710")
UFO_CHANNEL = os.getenv("UFO_CHANNEL", "calls")  # fallback when there's no call id


def channel_for(call_id: str | None) -> str:
    """One UFO conversation per phone call: the mid-call tasks and the end-of-call report for the same
    call share context (so nothing is done twice), and every call starts fresh and fast."""
    safe = "".join(c for c in (call_id or "").lower() if c.isalnum() or c == "-")[:48]
    return f"call-{safe}" if safe else UFO_CHANNEL


async def send_task(text: str, call_id: str | None) -> str | None:
    """Post a task to UFO in this call's conversation; returns the turn id (never raises)."""
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            r = await client.post(
                f"{UFO_URL}/surface/ufo/{channel_for(call_id)}",
                headers={**_headers(), "x-ufo-send": "1", "x-ufo-send-id": str(uuid.uuid4())},
                content=text.encode(),
            )
        r.raise_for_status()
    except Exception as e:
        logger.error("UFO hand-off failed: %r", e)
        return None
    parts = r.text.strip().split("\t")  # sent <turn_id> <opened_run> <arrival_id>
    return parts[1] if len(parts) > 1 and parts[0] == "sent" else None
UFO_TOKEN_FILE = Path(os.getenv("UFO_TOKEN_FILE", Path.home() / ".ufoctl" / "token"))
from dotenv import load_dotenv as _load_dotenv

_load_dotenv(Path(__file__).resolve().parent / ".env")  # SPEAKER_NAME etc. must be set before use, whoever imports us
SPEAKER = os.getenv("SPEAKER_NAME", "Michael")
MY_EMAIL = os.getenv("SPEAKER_EMAIL", "")  # set SPEAKER_EMAIL in .env

TASK = """A phone call just ended (call id {call_id}). {who_called}
Call ended: {ended_reason}. Duration: {duration}.

Summary: {summary}

Transcript:
{transcript}

Follow up for {speaker} with your river_* tools. Be quick: visible results first, then stop.
1. Duplicates: only what a "Live phone call {call_id}" task did earlier in THIS conversation counts. Drafts or
   notes from other calls (e.g. rehearsals) don't: this call still gets its own note and draft.
2. Only if the call is unclear about who this is or what was agreed: one river_situation(caller=<their number>)
   lookup. Otherwise skip lookups.
3. river_remember (slug calls/{date}-{slug}): who called, what was said or agreed, open follow-ups.
4. If anything was promised or requested (documents, a quote, a time change, an intro), river_draft_email for
   it. Never send. Unknown recipient address: send the draft to {my_email} with subject "[For <name>] ...".
5. Don't invent facts that aren't in the call.
6. Write every draft and note as {speaker} and sign emails "{speaker}". {my_email} is only the inbox drafts land in.
Reply with at most 3 short lines on what you did."""


def build_task(report: dict) -> str:
    msg = report.get("message", report)
    call = msg.get("call") or {}
    customer = call.get("customer") or msg.get("customer") or {}
    number = customer.get("number") or "an unknown number"
    name = (msg.get("caller_name") or customer.get("name") or "").strip()
    caller = f"{name} ({number})" if name else number
    analysis = msg.get("analysis") or {}
    artifact = msg.get("artifact") or {}
    transcript = msg.get("transcript") or artifact.get("transcript") or "(no transcript)"
    duration = msg.get("durationSeconds") or msg.get("duration_seconds")
    started = msg.get("startedAt") or call.get("createdAt") or ""
    meta = (call.get("assistantOverrides") or {}).get("metadata") or call.get("metadata") or {}
    if meta.get("direction") == "outbound" or call.get("type") == "outboundPhoneCall":
        name = name or (meta.get("caller_name") or "")
        caller = f"{name} ({number})" if name else number
        who_called = (f"{SPEAKER}'s AI voice agent called {caller} as {SPEAKER}"
                      f"{' to say: ' + meta['reason'] if meta.get('reason') else ''}.")
    else:
        who_called = f"{caller} called {SPEAKER}; {SPEAKER}'s AI voice agent answered as {SPEAKER}."
    return TASK.format(
        who_called=who_called, speaker=SPEAKER, my_email=MY_EMAIL, call_id=call.get("id") or "unknown",
        ended_reason=msg.get("endedReason") or "unknown",
        duration=f"{round(duration)} s" if isinstance(duration, (int, float)) else "unknown",
        summary=analysis.get("summary") or msg.get("summary") or "(none)",
        transcript=transcript,
        date=(started[:10] or "call"),
        slug=(name.split()[0].lower() if name else "".join(c for c in number if c.isdigit())[-4:] or "caller")
        + (f"-{''.join(c for c in str(call.get('id') or '') if c.isalnum())[:8]}" if call.get("id") else ""),
    )


def _headers() -> dict:
    return {"authorization": f"Bearer {UFO_TOKEN_FILE.read_text().strip()}"}


async def hand_off_call(report: dict) -> str | None:
    """Queue the follow-up in UFO (in this call's conversation); returns its turn id or None."""
    msg = report.get("message", report)
    call_id = (msg.get("call") or {}).get("id")
    turn_id = await send_task(build_task(report), call_id)
    logger.info("UFO follow-up for call %s queued: %s", call_id, turn_id)
    return turn_id


async def read_result(call_id: str | None = None, timeout_s: float = 240) -> str:
    """Tail a call's latest UFO turn until it finishes; returns its final text."""
    async with httpx.AsyncClient(timeout=timeout_s) as client:
        r = await client.post(f"{UFO_URL}/surface/ufo/{channel_for(call_id)}", headers=_headers(), content=b"")
    for line in r.text.splitlines():
        if line.startswith("frame\tterminal\t"):
            import json

            payload = line.split("\t", 2)[2]
            try:  # raw_decode: the frame can carry more tab-separated data after the JSON
                return json.JSONDecoder().raw_decode(payload)[0].get("text", "")
            except json.JSONDecodeError:
                m = re.search(r'"text":"(.*?)","incomplete_reason"', payload, re.S)
                return m.group(1).replace('\\"', '"').replace("\\n", "\n") if m else payload
    return r.text


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print(asyncio.run(read_result(sys.argv[1] if len(sys.argv) > 1 else None)))
