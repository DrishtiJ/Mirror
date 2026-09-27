"""What the phone owner (SPEAKER_NAME) is doing right now, for the agent answering their phone.

    situation_for_caller(caller_number) -> {"caller_name": str | None, "notes": str}

Sources, all through GBrain (gbrain.io):
- location: the `location/current` page, written by location_server.py from phone GPS pings
- next stop and why: Google Calendar (gcal_list_events + gcal_get_event)
- who's calling: the caller's number matched against calendar guests, Gmail and GBrain memory
- ETA: OpenStreetMap geocoding (Nominatim) + OSRM driving route from the phone's position

Caller-independent parts (location, calendar, ETA) are refreshed in the background every
REFRESH_S, so a call only waits on the caller lookup.

    uv run situation.py [+14155550123]     # print what the agent would know right now
"""

import asyncio
import html
import json
import logging
import math
import os
import re
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

import httpx
from pathlib import Path

import gbrain_context as gb

logger = logging.getLogger("situation")

from dotenv import load_dotenv as _load_dotenv

_load_dotenv(Path(__file__).resolve().parent / ".env")  # SPEAKER_NAME etc. must be set before use, whoever imports us
SPEAKER = os.getenv("SPEAKER_NAME", "Michael")
LOCATION_SLUG = "location/current"
REFRESH_S = float(os.getenv("SITUATION_REFRESH_S", "60"))
# Whole situation_for_caller stays under this (the Vapi server caps us at 2.5 s).
TOTAL_BUDGET_S = float(os.getenv("SITUATION_TOTAL_BUDGET_S", "2.2"))
LOCATION_MAX_AGE = timedelta(hours=float(os.getenv("GBRAIN_LOCATION_MAX_AGE_H", "6")))
HTTP_HEADERS = {"User-Agent": "river-voice-agent/0.1 (YC hackathon demo)"}


# --- small helpers ------------------------------------------------------------------------


def _json(text: str) -> Any:
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return None


async def _call(session, tool: str, args: dict) -> str:
    try:
        result = await session.call_tool(tool, args)
    except Exception as e:
        logger.warning("GBrain %s failed: %r", tool, e)
        return ""
    text = gb.result_text(result)
    if result.isError:
        logger.warning("GBrain %s returned an error: %s", tool, text)
        return ""
    return text


def _parse_time(value: Any, tz) -> datetime | None:
    """Google-style {dateTime|date} dicts or plain ISO strings."""
    if isinstance(value, dict):
        value = value.get("dateTime") or value.get("date_time") or value.get("date")
    if not isinstance(value, str) or not value:
        return None
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return dt.astimezone(tz) if dt.tzinfo else dt.replace(tzinfo=tz)


def _first(d: dict, *keys: str) -> Any:
    return next((d[k] for k in keys if d.get(k) not in (None, "", [])), None)


def _clock(dt: datetime) -> str:
    return f"{dt:%-I:%M %p}".replace(":00 ", " ")


def _duration(minutes: float) -> str:
    m = round(minutes)
    if m < 60:
        return f"{m} minutes"
    h, m = divmod(m, 60)
    hours = "an hour" if h == 1 else f"{h} hours"
    return hours if m < 5 else f"{hours} and {m} minutes"


def digits(number: str | None) -> str:
    return re.sub(r"\D", "", number or "")[-10:]


# --- location -----------------------------------------------------------------------------


@dataclass
class Fix:
    at: datetime
    lat: float
    lon: float
    speed_mps: float | None = None
    motion: str = ""
    place: str = ""

    @property
    def mph(self) -> float | None:
        return None if self.speed_mps is None or self.speed_mps < 0 else self.speed_mps * 2.23694

    def describe(self, now: datetime) -> str:
        mph = self.mph
        if self.motion in ("driving", "automotive") or (mph is not None and mph >= 15):
            doing = f"driving, about {round(mph)} miles an hour" if mph else "driving"
        elif self.motion in ("walking", "running") or (mph is not None and mph >= 2):
            doing = "walking"
        elif self.motion == "cycling":
            doing = "biking"
        else:
            doing = "not moving"
        where = f" on {self.place}" if self.place and doing.startswith("driving") else (
            f" near {self.place}" if self.place else "")
        age = max(0, int((now - self.at).total_seconds() // 60))
        when = "just now" if age < 2 else f"{age} minutes ago"
        return f"{doing}{where} (phone GPS, {when})"


def parse_location_page(text: str, now: datetime) -> Fix | None:
    """The page written by location_server.py: frontmatter/JSON with the latest fix."""
    data = _json(text) or {}
    content = data.get("content") or data.get("compiled_truth") or text
    m = re.search(r"```json\s*(\{.*?\})\s*```", content, re.S)
    fix = _json(m.group(1)) if m else None
    if not isinstance(fix, dict):
        return None
    at = _parse_time(fix.get("at"), now.tzinfo)
    if at is None or now - at > LOCATION_MAX_AGE:
        return None
    return Fix(at=at, lat=float(fix["lat"]), lon=float(fix["lon"]), speed_mps=fix.get("speed_mps"),
               motion=fix.get("motion") or "", place=fix.get("place") or "")


# --- calendar -----------------------------------------------------------------------------


@dataclass
class Event:
    id: str
    title: str
    start: datetime
    end: datetime | None
    location: str = ""
    description: str = ""
    guests: list[dict] = field(default_factory=list)

    def guest_names(self) -> list[str]:
        out = []
        for g in self.guests:
            if g.get("self"):
                continue
            name = g.get("displayName") or g.get("name") or g.get("email", "").split("@")[0]
            if name:
                out.append(name)
        return out


def parse_events(text: str, tz) -> list[Event]:
    data = _json(text)
    raw = data.get("events") if isinstance(data, dict) else data
    events = []
    for e in raw or []:
        start = _parse_time(_first(e, "start", "start_time", "starts_at", "when"), tz)
        if start is None:
            continue
        events.append(Event(
            id=str(_first(e, "id", "event_id") or ""),
            title=str(_first(e, "summary", "title", "name") or "an event"),
            start=start,
            end=_parse_time(_first(e, "end", "end_time", "ends_at"), tz),
            location=str(_first(e, "location", "place", "where") or ""),
            description=str(_first(e, "description", "notes") or ""),
            guests=list(_first(e, "attendees", "guests") or []) if isinstance(_first(e, "attendees", "guests"), list) else [],
        ))
    return sorted(events, key=lambda ev: ev.start)


def merge_full_event(ev: Event, text: str) -> Event:
    full = _json(text)
    if isinstance(full, dict):
        full = full.get("event", full)
        ev.description = str(_first(full, "description", "notes") or ev.description)
        ev.location = str(_first(full, "location", "place") or ev.location)
        guests = _first(full, "attendees", "guests")
        if isinstance(guests, list):
            ev.guests = guests
    return ev


# --- routing ------------------------------------------------------------------------------

_geocode_cache: dict[str, tuple[float, float] | None] = {}


async def geocode(client: httpx.AsyncClient, place: str) -> tuple[float, float] | None:
    """Calendar locations often lead with a venue name ("Blue Bottle Coffee, 300 Webster St, ...");
    try the whole string, then drop leading comma parts until OpenStreetMap finds it."""
    if place in _geocode_cache:
        return _geocode_cache[place]
    parts = [p.strip() for p in place.split(",") if p.strip()]
    candidates = [", ".join(parts[i:]) for i in range(max(1, len(parts) - 1))]
    for q in candidates:
        try:
            r = await client.get("https://nominatim.openstreetmap.org/search",
                                 params={"q": q, "format": "json", "limit": 1})
            hits = r.json()
        except Exception as e:
            logger.warning("geocode %r failed: %r", q, e)
            return None
        if hits:
            _geocode_cache[place] = (float(hits[0]["lat"]), float(hits[0]["lon"]))
            return _geocode_cache[place]
    logger.warning("geocode found nothing for %r", place)
    _geocode_cache[place] = None
    return None


async def reverse_geocode(client: httpx.AsyncClient, lat: float, lon: float) -> str:
    try:
        r = await client.get("https://nominatim.openstreetmap.org/reverse",
                             params={"lat": lat, "lon": lon, "format": "json", "zoom": 16})
        a = r.json().get("address", {})
        road = a.get("road") or a.get("highway") or ""
        town = a.get("city") or a.get("town") or a.get("village") or a.get("suburb") or ""
        return ", ".join(x for x in (road, town) if x)
    except Exception as e:
        logger.warning("reverse geocode failed: %r", e)
        return ""


async def drive_time(client: httpx.AsyncClient, frm: tuple[float, float], to: tuple[float, float]):
    """(minutes, miles) by car, or a straight-line estimate if OSRM is unreachable."""
    try:
        r = await client.get(
            f"https://router.project-osrm.org/route/v1/driving/{frm[1]},{frm[0]};{to[1]},{to[0]}",
            params={"overview": "false"})
        route = r.json()["routes"][0]
        return route["duration"] / 60, route["distance"] / 1609.34
    except Exception as e:
        logger.warning("OSRM failed, using straight-line estimate: %r", e)
        miles = _haversine_miles(frm, to)
        return miles / 30 * 60 * 1.3, miles


def _haversine_miles(a: tuple[float, float], b: tuple[float, float]) -> float:
    la1, lo1, la2, lo2 = map(math.radians, (*a, *b))
    h = math.sin((la2 - la1) / 2) ** 2 + math.cos(la1) * math.cos(la2) * math.sin((lo2 - lo1) / 2) ** 2
    return 3958.8 * 2 * math.asin(math.sqrt(h))


# --- the snapshot -------------------------------------------------------------------------


@dataclass
class Snapshot:
    now: datetime
    fix: Fix | None = None
    events: list[Event] = field(default_factory=list)  # today's remaining + next 24h
    next_event: Event | None = None
    eta: tuple[float, float] | None = None  # minutes, miles to next_event
    built_at: float = field(default_factory=time.monotonic)


async def build_snapshot() -> Snapshot:
    now = gb.local_now()
    snap = Snapshot(now=now)
    if not gb.is_configured():
        return snap
    async with gb.gbrain_session() as session, httpx.AsyncClient(timeout=4, headers=HTTP_HEADERS) as http:
        page, cal = await asyncio.gather(
            _call(session, "get_page", {"slug": LOCATION_SLUG, "include_content": True}),
            _call(session, "gcal_list_events", {
                "time_min": (now - timedelta(minutes=30)).isoformat(),
                "time_max": (now + timedelta(hours=24)).isoformat(), "limit": 100}),
        )
        snap.fix = parse_location_page(page, now) if page else None
        snap.events = [e for e in parse_events(cal, now.tzinfo) if (e.end or e.start) > now]
        # the current or next event: in progress (not yet ended) counts, so its venue and ETA stay in the notes
        snap.next_event = snap.events[0] if snap.events else None
        if snap.next_event and snap.next_event.id:
            snap.next_event = merge_full_event(
                snap.next_event, await _call(session, "gcal_get_event", {"id": snap.next_event.id}))
        if snap.fix and not snap.fix.place:
            snap.fix.place = await reverse_geocode(http, snap.fix.lat, snap.fix.lon)
        if snap.fix and snap.next_event and snap.next_event.location:
            dest = await geocode(http, snap.next_event.location)
            if dest:
                snap.eta = await drive_time(http, (snap.fix.lat, snap.fix.lon), dest)
    return snap


def render(snap: Snapshot, caller_name: str | None, caller_facts: list[str]) -> str:
    now = snap.now
    lines = [f"It's {now:%A, %B %-d, %Y}, {_clock(now)} (today is {now:%Y-%m-%d})."]
    if snap.fix:
        lines.append(f"You're {snap.fix.describe(now)}.")
    ev = snap.next_event
    if ev:
        started = ev.start <= now
        starts = f"started at {_clock(ev.start)}" if started else f"at {_clock(ev.start)}"
        where = f" at {ev.location}" if ev.location else ""
        lines.append(f"{'Right now' if started else 'Your next plan'}: {ev.title}, {starts}{where}.")
        guests = ev.guest_names()
        if guests:
            lines.append(f"With: {', '.join(guests)}.")
        if ev.description.strip():
            lines.append(f"About it: {' '.join(ev.description.split())}")
        if snap.eta:
            minutes, miles = snap.eta
            arrive = now + timedelta(minutes=minutes)
            slack = (ev.start - arrive).total_seconds() / 60
            if ev.start <= now:
                timing = f"so about {round((arrive - ev.start).total_seconds() / 60)} minutes late (it already started)"
            elif slack < 0:
                timing = f"about {round(-slack)} minutes late"
            elif slack <= 10:
                timing = "right on time"
            else:
                timing = f"with about {_duration(slack)} to spare"
            lines.append(f"You're about {round(minutes)} minutes away from there by car ({miles:.0f} miles), arriving around {_clock(arrive)}, {timing}.")
    later = [e for e in snap.events if e is not ev]
    if later:
        lines.append("Later today: " + "; ".join(f"{_clock(e.start)} {e.title}" for e in later) + ".")
    if caller_name:
        lines.append(f"The caller is {caller_name}.")
    lines.extend(caller_facts)
    return "\n".join(lines)


# --- caller lookup ------------------------------------------------------------------------


def _number_variants(number: str) -> list[str]:
    d = digits(number)
    if len(d) != 10:
        return [number] if number else []
    return [f"{d[:3]}-{d[3:6]}-{d[6:]}", f"({d[:3]}) {d[3:6]}-{d[6:]}", f"{d[:3]}.{d[3:6]}.{d[6:]}", d]


def _name_from_header(value: str) -> str:
    m = re.match(r'\s*"?([^"<]+?)"?\s*<', value or "")
    return m.group(1).strip() if m else ""


def local_contact(caller_number: str | None) -> str | None:
    """Instant match against DEMO_CONTACTS_JSON ({"+14155550123": "Dana Lee"})."""
    d = digits(caller_number)
    contacts = _json(os.getenv("DEMO_CONTACTS_JSON", "")) or {}
    return next((n for num, n in contacts.items() if d and digits(num) == d), None)


def meeting_fact(name: str | None, snap: Snapshot) -> list[str]:
    ev = snap.next_event
    if not (name and ev):
        return []
    first = name.split()[0].lower()
    on_it = any(first in g.lower() for g in ev.guest_names()) or re.search(
        rf"\b{re.escape(first)}\b", f"{ev.title} {ev.description}", re.I)
    return [f"{name} is part of your next plan ({ev.title}), so they're probably calling about it."] if on_it else []


async def lookup_caller(caller_number: str | None, snap: Snapshot, known_name: str | None = None) -> tuple[str | None, list[str]]:
    """Best-effort enrichment from Gmail and GBrain memory (network; run under a timeout)."""
    d = digits(caller_number)
    name = known_name
    facts: list[str] = []
    if not d or not gb.is_configured():
        return name, facts
    query = " OR ".join(f'"{v}"' for v in _number_variants(caller_number))
    async with gb.gbrain_session() as session:
        mail, memory = await asyncio.gather(
            _call(session, "gmail_search", {"query": query, "limit": 5}),
            _call(session, "search", {"query": d, "limit": 5}),
        )
    msgs = (_json(mail) or {}).get("messages") or (_json(mail) or {}).get("results") or []
    if not name:
        for m in msgs:
            name = _name_from_header(str(_first(m, "from", "sender") or "")) or None
            if name:
                break
    # A number in an email body is weak evidence (it can be your own signature), so only use
    # mail to identify callers we don't already know, and only when the sender gave us a name.
    if msgs and not known_name and name:
        m = msgs[0]
        subject = html.unescape(str(_first(m, "subject") or ""))
        snippet = html.unescape(str(_first(m, "snippet", "preview") or ""))
        facts.append(f"Last email that mentions their number: \"{subject}\" — {snippet}".strip())
    for hit in _json(memory) or []:
        text = str(hit.get("chunk_text") or "")
        # only notes that literally contain the number, not "semantically similar" pages
        if d in digits(text) or any(v in text for v in _number_variants(caller_number)):
            facts.append(f"From your notes ({hit.get('title') or hit.get('slug')}): {' '.join(text.split())}")
    return name, facts


# --- public API ---------------------------------------------------------------------------

_snapshot: Snapshot | None = None
_first_build: asyncio.Future | None = None
_refresher: asyncio.Task | None = None


async def _refresh_forever() -> None:
    global _snapshot
    while True:
        try:
            _snapshot = await build_snapshot()
        except Exception as e:
            logger.warning("situation refresh failed: %r", e)
        if _first_build and not _first_build.done():
            _first_build.set_result(None)
        await asyncio.sleep(REFRESH_S)


def prewarm() -> None:
    """Start the background refresher. Call once at server startup so the first call is warm."""
    global _refresher, _first_build
    if _refresher is None or _refresher.done():
        _first_build = asyncio.get_running_loop().create_future()
        _refresher = asyncio.create_task(_refresh_forever())


async def current_snapshot(timeout: float) -> Snapshot:
    prewarm()
    if _snapshot is None:
        try:
            await asyncio.wait_for(asyncio.shield(_first_build), timeout)
        except TimeoutError:
            logger.warning("situation snapshot not ready within %.1fs; answering with time only", timeout)
    return _snapshot or Snapshot(now=gb.local_now())


async def situation_for_caller(caller_number: str | None) -> dict:
    deadline = time.monotonic() + TOTAL_BUDGET_S
    known = local_contact(caller_number)  # instant; never lost to a slow lookup
    snap = await current_snapshot(timeout=TOTAL_BUDGET_S * 0.6)
    name, facts = known, []
    remaining = deadline - time.monotonic()
    if remaining > 0.2:
        try:
            name, facts = await asyncio.wait_for(lookup_caller(caller_number, snap, known), remaining)
        except Exception as e:  # incl. timeout: keep the local name, skip enrichment
            logger.warning("caller enrichment skipped: %r", e)
    name = name or known
    snap.now = gb.local_now()
    return {"caller_name": name, "notes": render(snap, name, facts + meeting_fact(name, snap))}


LATE_THRESHOLD_MIN = float(os.getenv("LATE_THRESHOLD_MIN", "5"))
LATE_WINDOW_MIN = float(os.getenv("LATE_WINDOW_MIN", "90"))  # only watch events starting this soon


def late_status(snap: Snapshot) -> dict | None:
    """Deterministic lateness: arrival (now + drive time) vs the next event's start.
    None when there's nothing to judge (no fresh location, no upcoming event with a place)."""
    ev, now = snap.next_event, snap.now
    if not (ev and snap.eta and snap.fix) or ev.start <= now:
        return None
    if (ev.start - now).total_seconds() / 60 > LATE_WINDOW_MIN:
        return None
    minutes, _ = snap.eta
    late_by = (now + timedelta(minutes=minutes) - ev.start).total_seconds() / 60
    return {"late": late_by > LATE_THRESHOLD_MIN, "late_by_min": round(late_by), "eta_min": round(minutes),
            "event_id": ev.id, "event": ev.title, "start": ev.start.isoformat(), "location": ev.location}


def late_line(status: dict | None) -> str:
    """One stable line for UFO's monitor probe: it wakes the agent only when this changes."""
    if not status or not status["late"]:
        return "ok"
    # coarse 5-minute buckets so a few seconds of drift doesn't re-fire the monitor
    bucket = 5 * max(1, round(status["late_by_min"] / 5))
    return f"LATE ~{bucket} min · {status['event']} · {status['start']} · id {status['event_id']}"


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))
    logging.basicConfig(level=logging.WARNING)
    t0 = time.perf_counter()
    out = asyncio.run(situation_for_caller(sys.argv[1] if len(sys.argv) > 1 else None))
    print(f"[{time.perf_counter() - t0:.2f}s] caller_name={out['caller_name']!r}\n{out['notes']}")
