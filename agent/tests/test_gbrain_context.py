"""Offline tests for the GBrain context layer (no network, no tokens)."""

import asyncio
from datetime import datetime
from types import SimpleNamespace
from zoneinfo import ZoneInfo


import gbrain_context as gb


def tool(name, **props):
    return SimpleNamespace(name=name, description="", inputSchema={"type": "object", "properties": props})


TOOLS = [
    tool("gmail_search", query={"type": "string"}, max_results={"type": "integer"}),
    tool("gcal_list_events", timeMin={"type": "string"}, timeMax={"type": "string"}, limit={"type": "integer"}),
    tool("memory_search", q={"type": "string"}, top_k={"type": "integer"}),
    tool("exa_search", query={"type": "string"}),
]


def test_picks_tools_by_pattern():
    assert gb.pick_tool(TOOLS, "memory").name == "memory_search"
    assert gb.pick_tool(TOOLS, "calendar").name == "gcal_list_events"
    assert gb.pick_tool(TOOLS, "email").name == "gmail_search"


def test_pinned_tool_name_wins(monkeypatch):
    monkeypatch.setitem(gb.TOOL_PATTERNS, "memory", ("exa_search", r"memory"))
    assert gb.pick_tool(TOOLS, "memory").name == "exa_search"
    monkeypatch.setitem(gb.TOOL_PATTERNS, "memory", ("missing_tool", r"memory"))
    assert gb.pick_tool(TOOLS, "memory") is None


def test_args_follow_each_tools_schema():
    start = datetime(2026, 9, 27, tzinfo=ZoneInfo("America/Los_Angeles"))
    assert gb.build_args(TOOLS[2], query="Dana", limit=5) == {"q": "Dana", "top_k": 5}
    cal = gb.build_args(TOOLS[1], time_min=start, time_max=start, limit=10, query="ignored")
    assert cal == {"timeMin": start.isoformat(), "timeMax": start.isoformat(), "limit": 10}


def test_render_keeps_everything_and_skips_empty_sections():
    now = datetime(2026, 9, 27, 14, 5, tzinfo=ZoneInfo("America/Los_Angeles"))
    long_schedule = "\n".join(f"{h}:00 event {h}" for h in range(24)) * 20
    text = gb.PersonalContext(now=now, location="YC office, SF", schedule=long_schedule).render()
    assert "Sunday September 27, 2026, 2:05 PM PDT" in text
    assert "Where you are: YC office, SF" in text
    assert long_schedule in text  # nothing truncated
    assert "about them" not in text and "emails" not in text  # empty sections are skipped


def test_unconfigured_returns_time_and_env_location(monkeypatch, tmp_path):
    monkeypatch.setattr(gb, "TOKEN_DIR", tmp_path)
    monkeypatch.setenv("AGENT_LOCATION", "San Francisco")
    ctx = asyncio.run(gb.fetch_personal_context_safe("Michael", "Dana"))
    assert ctx.location == "San Francisco" and ctx.schedule == ""


def test_slow_gbrain_never_blocks_the_call(monkeypatch):
    async def hang(*_):
        await asyncio.sleep(10)

    monkeypatch.setattr(gb, "fetch_personal_context", hang)
    monkeypatch.setattr(gb, "FETCH_TIMEOUT_S", 0.05)
    ctx = asyncio.run(gb.fetch_personal_context_safe("Michael", "Dana"))
    assert ctx.schedule == "" and ctx.now is not None


def test_fetch_uses_session_and_fills_sections(monkeypatch, tmp_path):
    (tmp_path / "tokens.json").write_text("{}")
    monkeypatch.setattr(gb, "TOKEN_DIR", tmp_path)
    monkeypatch.setenv("AGENT_LOCATION", "Stale manual fallback")  # a fresh phone ping must win
    calls = []

    class FakeSession:
        async def list_tools(self):
            return SimpleNamespace(tools=TOOLS)

        async def call_tool(self, name, args):
            calls.append((name, args))
            text = {"memory_search": "note", "gcal_list_events": "3pm demo", "gmail_search": "re: demo"}[name]
            if name == "memory_search" and args["q"] == gb.LOCATION_TAG:
                text = f"{gb.LOCATION_TAG} {gb.local_now().isoformat()} | Pier 70, San Francisco | 37.76,-122.38"
            return SimpleNamespace(isError=False, content=[SimpleNamespace(text=text)], structuredContent=None)

    class FakeCM:
        async def __aenter__(self):
            return FakeSession()

        async def __aexit__(self, *a):
            return False

    monkeypatch.setattr(gb, "gbrain_session", lambda **_: FakeCM())
    ctx = asyncio.run(gb.fetch_personal_context("Michael", "Dana"))
    assert ctx.location == "Pier 70, San Francisco (from your phone, just now)"
    assert (ctx.schedule, ctx.caller_notes, ctx.caller_emails) == ("3pm demo", "note", "re: demo")
    assert ("memory_search", {"q": "Dana", "top_k": 10}) in calls
    assert ctx.sources == ["location", "calendar", "memory", "email"]


def test_file_token_storage_roundtrip(tmp_path):
    store = gb.FileTokenStorage(tmp_path)
    tok = gb.OAuthToken(access_token="a", token_type="Bearer", refresh_token="r")
    asyncio.run(store.set_tokens(tok))
    assert asyncio.run(store.get_tokens()).refresh_token == "r"
    assert oct((tmp_path / "tokens.json").stat().st_mode)[-3:] == "600"


def test_unknown_caller_skips_caller_lookups(monkeypatch, tmp_path):
    (tmp_path / "tokens.json").write_text("{}")
    monkeypatch.setattr(gb, "TOKEN_DIR", tmp_path)
    called = []

    async def fake_call(session, tools, kind, **kw):
        called.append((kind, kw.get("query")))
        return ""

    class FakeCM:
        async def __aenter__(self):
            return SimpleNamespace(list_tools=lambda: _tools())

        async def __aexit__(self, *a):
            return False

    async def _tools():
        return SimpleNamespace(tools=TOOLS)

    monkeypatch.setattr(gb, "gbrain_session", lambda **_: FakeCM())
    monkeypatch.setattr(gb, "call", fake_call)
    asyncio.run(gb.fetch_personal_context("Michael", "the caller"))
    assert [k for k, _ in called] == ["memory", "calendar"]


def test_merge_notes_keeps_dispatch_notes_and_drops_placeholder():
    now = datetime(2026, 9, 27, 14, 5, tzinfo=ZoneInfo("America/Los_Angeles"))
    ctx = gb.PersonalContext(now=now, location="SF")
    assert gb.merge_notes("No prior context.", ctx, default="No prior context.").startswith("Right now:")
    merged = gb.merge_notes("Following up on the demo.", ctx, default="No prior context.")
    assert merged.splitlines()[0] == "Following up on the demo." and "Where you are: SF" in merged


NOW = datetime(2026, 9, 27, 14, 0, tzinfo=ZoneInfo("America/Los_Angeles"))


def test_latest_location_picks_newest_fresh_ping():
    text = "\n".join([
        "LOCATION UPDATE 2026-09-27T09:00:00-07:00 | Home, Oakland | 37.8,-122.2",
        "some unrelated note that mentions a location",
        "LOCATION UPDATE 2026-09-27T13:20:00-07:00 | YC, San Francisco | 37.79,-122.39",
    ])
    assert gb.latest_location(text, NOW) == "YC, San Francisco (from your phone, 40 minutes ago)"


def test_stale_or_missing_ping_means_unknown_not_a_guess():
    assert gb.latest_location("LOCATION UPDATE 2026-09-26T08:00:00-07:00 | Home", NOW) == ""
    assert gb.latest_location("Michael is usually in SF", NOW) == ""
    assert gb.latest_location("LOCATION UPDATE not-a-date | Home", NOW) == ""



def test_refresh_only_when_near_expiry(monkeypatch, tmp_path):
    import json, os, time

    (tmp_path / "tokens.json").write_text(json.dumps({"access_token": "a", "refresh_token": "r", "expires_in": 3600}))
    (tmp_path / "client.json").write_text(json.dumps({"client_id": "c", "token_endpoint_auth_method": "none"}))
    monkeypatch.setattr(gb, "TOKEN_DIR", tmp_path)
    posted = []

    class FakeResp:
        is_error = False
        def json(self): return {"access_token": "new", "expires_in": 3600}

    class FakeClient:
        def __init__(self, *a, **k): pass
        async def __aenter__(self): return self
        async def __aexit__(self, *a): return False
        async def post(self, url, data):
            posted.append(data)
            return FakeResp()

    monkeypatch.setattr(gb.httpx, "AsyncClient", FakeClient)
    real_storage = gb.FileTokenStorage
    monkeypatch.setattr(gb, "FileTokenStorage", lambda: real_storage(tmp_path))
    assert asyncio.run(gb.refresh_tokens()) is False and posted == []  # fresh token: no call
    old = time.time() - 3500
    os.utime(tmp_path / "tokens.json", (old, old))
    assert asyncio.run(gb.refresh_tokens()) is True
    assert posted[0]["grant_type"] == "refresh_token" and posted[0]["refresh_token"] == "r"
    saved = json.loads((tmp_path / "tokens.json").read_text())
    assert saved["access_token"] == "new" and saved["refresh_token"] == "r"  # kept when not rotated
