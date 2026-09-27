"""Offline tests for situation.py and location_server.py parsing/rendering."""

import json
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import location_server
import situation as st

TZ = ZoneInfo("America/Los_Angeles")
NOW = datetime(2026, 9, 28, 12, 5, tzinfo=TZ)


def test_overland_and_flat_payloads_normalize():
    overland = {"locations": [
        {"geometry": {"coordinates": [-122.28, 37.79]}, "properties": {"timestamp": "2026-09-28T19:04:00Z", "speed": 25, "motion": ["driving"]}},
        {"geometry": {"coordinates": [-122.29, 37.80]}, "properties": {"timestamp": "2026-09-28T19:05:00Z", "speed": 26, "motion": ["driving"]}},
    ]}
    fixes = location_server.fixes_from_payload(overland)
    assert fixes[-1] == {"at": "2026-09-28T19:05:00Z", "lat": 37.80, "lon": -122.29, "speed_mps": 26, "motion": "driving"}
    flat = location_server.fixes_from_payload({"lat": "37.7", "lon": "-122.4", "speed_mps": 1.2})
    assert flat[0]["lat"] == 37.7 and flat[0]["speed_mps"] == 1.2


def test_location_page_roundtrip_and_staleness():
    fix = {"at": (NOW - timedelta(minutes=3)).isoformat(), "lat": 37.79, "lon": -122.28, "speed_mps": 25, "motion": "driving"}
    page = json.dumps({"content": location_server.page_content(fix)})
    parsed = st.parse_location_page(page, NOW)
    assert parsed.mph > 55 and parsed.describe(NOW).startswith("driving, about 56 miles an hour")
    assert "3 minutes ago" in parsed.describe(NOW)
    stale = dict(fix, at=(NOW - timedelta(hours=7)).isoformat())
    assert st.parse_location_page(json.dumps({"content": location_server.page_content(stale)}), NOW) is None


def test_parse_events_google_shape_and_next_event_render():
    cal = json.dumps({"events": [
        {"id": "e2", "summary": "Dinner", "start": {"dateTime": "2026-09-28T19:00:00-07:00"}},
        {"id": "e1", "summary": "Coffee with Dana and Priya", "start": {"dateTime": "2026-09-28T12:30:00-07:00"},
         "end": {"dateTime": "2026-09-28T13:30:00-07:00"}, "location": "Blue Bottle, 300 Webster St, Oakland"},
    ]})
    events = st.parse_events(cal, TZ)
    assert [e.id for e in events] == ["e1", "e2"]
    ev = st.merge_full_event(events[0], json.dumps({"description": "Intro to Priya re: seed round",
        "attendees": [{"email": "me@x.com", "self": True}, {"email": "dana@x.com", "displayName": "Dana"}, {"email": "priya@y.com"}]}))
    snap = st.Snapshot(now=NOW, fix=st.Fix(at=NOW, lat=37.79, lon=-122.28, speed_mps=25, motion="driving", place="I-880, Oakland"),
                       events=events, next_event=ev, eta=(12.4, 6.1))
    notes = st.render(snap, "Dana", [])
    assert "You're driving, about 56 miles an hour on I-880, Oakland (phone GPS, just now)." in notes
    assert "Your next plan: Coffee with Dana and Priya, at 12:30 PM at Blue Bottle, 300 Webster St, Oakland." in notes
    assert "With: Dana, priya." in notes and "About it: Intro to Priya re: seed round" in notes
    assert "about 12 minutes away from there by car (6 miles), arriving around 12:17 PM, with about 13 minutes to spare." in notes
    assert "Later today: 7 PM Dinner." in notes and "The caller is Dana." in notes


def test_late_arrival_is_called_out():
    ev = st.Event(id="e", title="Coffee", start=NOW + timedelta(minutes=5), end=None, location="X")
    notes = st.render(st.Snapshot(now=NOW, next_event=ev, events=[ev], eta=(15, 5)), None, [])
    assert "about 10 minutes late" in notes


def test_number_matching_helpers():
    assert st.digits("+1 (415) 555-0123") == "4155550123"
    assert "415-555-0123" in st._number_variants("+14155550123")
    assert st._name_from_header('"Dana Lee" <dana@x.com>') == "Dana Lee"


def test_timing_wording():
    ev = st.Event(id="e", title="Coffee", start=NOW + timedelta(minutes=90), end=None, location="X")
    notes = st.render(st.Snapshot(now=NOW, next_event=ev, events=[ev], eta=(7, 4)), None, [])
    assert "with about an hour and 23 minutes to spare" in notes
    ev.start = NOW + timedelta(minutes=12)
    assert "right on time" in st.render(st.Snapshot(now=NOW, next_event=ev, events=[ev], eta=(7, 4)), None, [])


def test_local_contact_survives_slow_enrichment(monkeypatch):
    import asyncio

    monkeypatch.setenv("DEMO_CONTACTS_JSON", '{"+18025550100": "Sam"}')
    ev = st.Event(id="e", title="Coffee with Sam and Priya", start=NOW + timedelta(hours=1), end=None)
    monkeypatch.setattr(st, "_snapshot", st.Snapshot(now=NOW, next_event=ev, events=[ev]))
    monkeypatch.setattr(st, "prewarm", lambda: None)
    monkeypatch.setattr(st, "TOTAL_BUDGET_S", 0.4)

    async def hang(*a, **k):
        await asyncio.sleep(5)

    monkeypatch.setattr(st, "lookup_caller", hang)
    out = asyncio.run(st.situation_for_caller("+1 (802) 555-0100"))
    assert out["caller_name"] == "Sam"
    assert "Sam is part of your next plan" in out["notes"]


def test_cold_snapshot_never_blows_the_budget(monkeypatch):
    import asyncio, time

    monkeypatch.setattr(st, "_snapshot", None)
    monkeypatch.setattr(st, "TOTAL_BUDGET_S", 0.3)

    async def run():
        st._first_build = asyncio.get_running_loop().create_future()  # never resolves
        monkeypatch.setattr(st, "prewarm", lambda: None)
        t0 = time.monotonic()
        out = await st.situation_for_caller(None)
        return time.monotonic() - t0, out

    elapsed, out = asyncio.run(run())
    assert elapsed < 0.5 and out["notes"].startswith("It's ")


def test_late_status_and_probe_line():
    ev = st.Event(id="ev1", title="Coffee", start=NOW + timedelta(minutes=10), end=None, location="X")
    fix = st.Fix(at=NOW, lat=1, lon=1, speed_mps=20, motion="driving")
    late = st.late_status(st.Snapshot(now=NOW, fix=fix, next_event=ev, events=[ev], eta=(22, 9)))
    assert late["late"] and late["late_by_min"] == 12 and late["event_id"] == "ev1"
    assert st.late_line(late).startswith("LATE ~10 min · Coffee · ")
    on_time = st.late_status(st.Snapshot(now=NOW, fix=fix, next_event=ev, events=[ev], eta=(8, 3)))
    assert on_time["late"] is False and st.late_line(on_time) == "ok"
    # no fresh location, or the event is far off -> nothing to judge
    assert st.late_status(st.Snapshot(now=NOW, next_event=ev, events=[ev], eta=(22, 9))) is None
    far = st.Event(id="e2", title="Dinner", start=NOW + timedelta(hours=5), end=None, location="X")
    assert st.late_status(st.Snapshot(now=NOW, fix=fix, next_event=far, events=[far], eta=(22, 9))) is None
    assert st.late_line(None) == "ok"


def test_in_progress_event_keeps_venue_and_eta():
    ev = st.Event(id="e", title="Coffee with Priya", start=NOW - timedelta(minutes=20), end=NOW + timedelta(minutes=40),
                  location="Sightglass Coffee, 270 7th St, San Francisco")
    notes = st.render(st.Snapshot(now=NOW, next_event=ev, events=[ev], eta=(7, 4)), None, [])
    assert "Right now: Coffee with Priya, started at 11:45 AM at Sightglass Coffee, 270 7th St, San Francisco." in notes
    assert "so about 27 minutes late (it already started)" in notes
    assert "Later" not in notes
