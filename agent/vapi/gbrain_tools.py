"""Live GBrain lookups the Vapi agent can call mid-conversation (all read-only).

Each tool returns a short plain-text answer for the model to speak from. Vapi posts
tool calls to /vapi/tool (vapi/fish_bridge.py), which dispatches here.
"""

import asyncio
import html
import json
import logging
import re
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import gbrain_context as gb  # noqa: E402  (other workstream's auth + session helpers)

log = logging.getLogger("uvicorn.error")

TOOL_TIMEOUT_S = 6.0
MAX_CHARS = 1200
WEEKDAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]


# --- helpers ------------------------------------------------------------------------------


def resolve_day(day: str, now: datetime) -> datetime:
    """'today' | 'tomorrow' | weekday name (next occurrence) | YYYY-MM-DD -> local midnight."""
    d = (day or "today").strip().lower()
    base = now.replace(hour=0, minute=0, second=0, microsecond=0)
    if d in ("today", "tonight"):
        return base
    if d == "tomorrow":
        return base + timedelta(days=1)
    for i, name in enumerate(WEEKDAYS):
        if name in d:
            ahead = (i - base.weekday()) % 7
            if "next" in d and ahead == 0:
                ahead = 7
            return base + timedelta(days=ahead)
    try:
        parsed = datetime.fromisoformat(d[:10])
    except ValueError:
        raise ValueError(f"Couldn't understand the day {day!r}. Use today, tomorrow, a weekday, or YYYY-MM-DD.") from None
    when = base.replace(year=parsed.year, month=parsed.month, day=parsed.day)
    if abs((when - base).days) > 180:  # models without a date anchor invent years (e.g. 2025-07-29)
        raise ValueError(f"{day} looks wrong; today is {base:%Y-%m-%d}. Use today, tomorrow, or a weekday.")
    return when


def _clock(value: str | None, tz) -> str:
    if not value:
        return ""
    try:
        t = datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(tz)
    except ValueError:
        return value
    return t.strftime("%-I:%M %p").replace(":00 ", " ")


def _json(text: str) -> Any:
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return None


def _clip(text: str) -> str:
    text = re.sub(r"\s+", " ", html.unescape(text)).strip()
    return text if len(text) <= MAX_CHARS else text[:MAX_CHARS] + "…"


async def _call(name: str, args: dict) -> str:
    async with gb.gbrain_session() as session:
        result = await session.call_tool(name, args)
    text = gb.result_text(result)
    if result.isError:
        raise RuntimeError(text[:300] or f"{name} failed")
    return text


# --- tools --------------------------------------------------------------------------------


async def check_calendar(day: str = "today") -> str:
    now = gb.local_now()
    start = resolve_day(day, now)
    raw = await _call("gcal_list_events", {
        "time_min": start.isoformat(), "time_max": (start + timedelta(days=1)).isoformat(), "limit": 25,
    })
    events = (_json(raw) or {}).get("events") or []
    label = start.strftime("%A %B %-d")
    if not events:
        return f"Nothing on your calendar {label}. You're free all day."
    lines = []
    for ev in events:
        s = ev.get("start") or {}
        e = ev.get("end") or {}
        when = "all day" if isinstance(s, dict) and s.get("date") and not s.get("dateTime") else (
            f"{_clock(s.get('dateTime') if isinstance(s, dict) else s, now.tzinfo)}"
            f" to {_clock(e.get('dateTime') if isinstance(e, dict) else e, now.tzinfo)}")
        place = f" at {ev['location']}" if ev.get("location") else ""
        lines.append(f"{when}: {ev.get('summary') or ev.get('title') or 'busy'}{place}")
    return _clip(f"{label}: " + "; ".join(lines) + ". Anything else that day is free.")


async def lookup_person(name: str) -> str:
    card = _json(await _call("entity", {"name": name})) or {}
    if card.get("found"):
        return _clip(json.dumps({k: v for k, v in card.items() if k not in ("protocol_version", "latency_ms")}))
    hits = await recall(name)
    return hits if not hits.startswith("Nothing") else f"You don't have anything saved about {name}."


async def recall(query: str) -> str:
    data = _json(await _call("recall", {"query": query, "limit": 5})) or {}
    facts = [f.get("text") or f.get("fact") or "" for f in data.get("facts") or []]
    chunks = [
        f"{r.get('title') or r.get('slug')}: {r.get('chunk') or r.get('chunk_text') or ''}"
        for r in data.get("results") or []
        if not str(r.get("slug", "")).startswith("location/")  # live location is already in the notes
    ]
    found = [x for x in facts + chunks if x.strip()]
    return _clip(" | ".join(found[:5])) if found else f"Nothing in your notes about {query}."


async def search_email(query: str) -> str:
    data = _json(await _call("gmail_search", {"query": query, "limit": 3})) or {}
    msgs = data.get("messages") or []
    if not msgs:
        return f"No emails about {query}."
    out = []
    for m in msgs[:3]:
        sender = re.sub(r"\s*<.*?>", "", str(m.get("from") or ""))
        out.append(f"{m.get('date', '')[:16]} from {sender}: \"{m.get('subject', '')}\" — {m.get('snippet') or m.get('preview') or ''}")
    return _clip(" | ".join(out))


async def open_loops(person: str = "") -> str:
    args: dict[str, Any] = {"limit": 5}
    if person:
        args["counterparty"] = person
    data = _json(await _call("open_loops", args)) or {}
    groups = data.get("groups") or []
    if not groups:
        who = f" with {person}" if person else ""
        return f"No open threads or promises{who} on file."
    return _clip(json.dumps(groups))


_UFO_TASKS: set[asyncio.Task] = set()


async def hand_off_task(task: str, _call: dict | None = None) -> str:
    """Queue a real-world task in UFO right now and return immediately (UFO works in the background)."""
    import post_call  # other workstream: send_task(text, call_id) -> turn id, never raises

    call = _call or {}
    who = ((call.get("customer") or {}).get("number")) or "unknown number"
    meta = (call.get("assistantOverrides") or {}).get("metadata") or {}
    text = (
        f"Live phone call {call.get('id', '')} with {meta.get('caller_name') or who} is still in progress. "
        f"During the call, {post_call.SPEAKER} agreed to this task: {task}\n"
        f"Do it now with your river_* tools (drafts only, never send). The end-of-call report for this "
        f"same call will arrive later; don't do this task twice. Only this call's own conversation counts "
        f"as already done: drafts or notes from other calls don't, so do the task. Be quick: at most one "
        f"lookup. If you don't have a recipient's address, draft it to {post_call.MY_EMAIL} with subject "
        f"'[For <name>] ...'."
    )

    async def send() -> None:
        # Same per-call UFO conversation as the end-of-call report, so UFO can dedupe.
        turn = await post_call.send_task(text, call.get("id"))
        log.info("UFO task queued for call %s: %r -> turn %s", call.get("id"), task[:80], turn)

    t = asyncio.create_task(send())
    _UFO_TASKS.add(t)
    t.add_done_callback(_UFO_TASKS.discard)
    return "Queued. It'll be handled in the background; just confirm briefly and move on."


TOOLS = {
    "check_calendar": check_calendar,
    "lookup_person": lookup_person,
    "recall": recall,
    "search_email": search_email,
    "open_loops": open_loops,
    "hand_off_task": hand_off_task,
}
NEEDS_GBRAIN = {"check_calendar", "lookup_person", "recall", "search_email", "open_loops"}


async def run_tool(name: str, args: dict, call: dict | None = None) -> str:
    fn = TOOLS.get(name)
    args = {k: v for k, v in args.items() if k != "bot_say"}  # Vapi strips it; be safe if it doesn't
    if fn is None:
        return f"Unknown tool {name}."
    if name in NEEDS_GBRAIN and not gb.is_configured():
        return "Your calendar and notes aren't connected right now. Say you're not sure."
    if name == "hand_off_task":
        args = {**args, "_call": call}
    try:
        return await asyncio.wait_for(fn(**args), TOOL_TIMEOUT_S)
    except asyncio.TimeoutError:
        return "The lookup took too long. Say you'll have to check later."
    except (TypeError, ValueError) as e:
        return f"Lookup failed: {e}"
    except Exception as e:
        log.exception("tool %s failed", name)
        return f"Lookup failed ({type(e).__name__}). Say you're not sure offhand."


# --- Vapi tool definitions -------------------------------------------------------------------


BOT_SAY = {
    "type": "string",
    "description": (
        "Your spoken stall while you look this up, 3 to 7 words, like a real person thinking out loud on the "
        "phone. Start with a reaction word (oh, hm, yeah, uh, ok, wait) and echo the specific thing they asked "
        "about, in fragments, lowercase, casual. Don't narrate what you're doing: never say checking, looking up, "
        "calendar, schedule, email, notes, loops, or details. Different wording from any earlier stall. No trailing punctuation."
    ),
}


def _fn(name: str, description: str, params: dict, required: list[str]) -> dict:
    # bot_say: Vapi's reserved arg -- it speaks the model's own line before the tool runs.
    return {
        "name": name,
        "description": description,
        "parameters": {
            "type": "object",
            "properties": {**params, "bot_say": BOT_SAY},
            "required": [*required, "bot_say"],
        },
    }


SCHEMAS = [
    _fn("check_calendar",
        "See what's on your calendar for a day, before answering about plans or agreeing to a time.",
        {"day": {"type": "string", "description": "today, tomorrow, a weekday like 'tuesday', or YYYY-MM-DD"}},
        ["day"]),
    _fn("lookup_person",
        "Pull up what you know about a person or company the caller mentions.",
        {"name": {"type": "string", "description": "The person or company name"}},
        ["name"]),
    _fn("recall",
        "Search your notes and memory for something the caller brings up: a past conversation, a project, a detail.",
        {"query": {"type": "string", "description": "What to look up"}},
        ["query"]),
    _fn("search_email",
        "Search your email. Use this when they ask whether someone sent, emailed, or forwarded something, or about an email thread or intro.",
        {"query": {"type": "string", "description": "Search terms, e.g. a name or subject"}},
        ["query"]),
    _fn("open_loops",
        "Check promises and follow-ups: what you told someone you'd do, or what they're waiting on from you. Not for documents someone sent you (use search_email).",
        {"person": {"type": "string", "description": "Who, or empty for everyone"}},
        []),
]


SCHEMAS.append(_fn(
    "hand_off_task",
    "Get something done after agreeing to it on the call: draft or send an email or a document, "
    "follow up with someone, schedule or move something. Call it right away with everything you know; "
    "don't ask clarifying questions first, the background assistant has the call context.",
    {"task": {"type": "string", "description": "The task in one or two sentences, with names, what, and when"}},
    ["task"],
))


def vapi_tools(server_url: str) -> list[dict]:
    """Inline model.tools for the Vapi assistant. The empty request-start stops Vapi's canned filler;
    the model's own bot_say line is spoken instead."""
    return [
        {
            "type": "function",
            "function": s,
            "server": {"url": server_url, "timeoutSeconds": 10},
            "messages": [{"type": "request-start", "content": ""}],
        }
        for s in SCHEMAS
    ]
