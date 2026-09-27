"""Offline tests for vapi/gbrain_tools.py (no GBrain calls)."""

import asyncio
from datetime import datetime
from zoneinfo import ZoneInfo

import pytest

from vapi import gbrain_tools as t

SUNDAY = datetime(2026, 9, 27, 14, 40, tzinfo=ZoneInfo("America/Los_Angeles"))


@pytest.mark.parametrize(
    "day, expected",
    [
        ("today", "2026-09-27"),
        ("tomorrow", "2026-09-28"),
        ("tuesday", "2026-09-29"),
        ("Tuesday afternoon", "2026-09-29"),
        ("sunday", "2026-09-27"),
        ("next sunday", "2026-10-04"),
        ("2026-10-15", "2026-10-15"),
    ],
)
def test_resolve_day(day, expected):
    assert t.resolve_day(day, SUNDAY).date().isoformat() == expected


def test_resolve_day_rejects_garbage():
    with pytest.raises(ValueError):
        t.resolve_day("whenever", SUNDAY)
    with pytest.raises(ValueError, match="today is 2026-09-27"):
        t.resolve_day("2025-07-29", SUNDAY)  # a model's hallucinated year


def test_every_tool_requires_bot_say_and_suppresses_canned_filler():
    tools = t.vapi_tools("https://x/vapi/tool")
    assert {x["function"]["name"] for x in tools} == set(t.TOOLS)
    for x in tools:
        params = x["function"]["parameters"]
        assert "bot_say" in params["properties"] and "bot_say" in params["required"]
        assert x["messages"] == [{"type": "request-start", "content": ""}]
        assert x["server"]["url"] == "https://x/vapi/tool"


def test_run_tool_strips_bot_say_and_reports_unknown(monkeypatch):
    seen = {}

    async def fake(day: str = "today") -> str:
        seen["day"] = day
        return "ok"

    monkeypatch.setitem(t.TOOLS, "check_calendar", fake)
    monkeypatch.setattr(t.gb, "is_configured", lambda: True)
    assert asyncio.run(t.run_tool("check_calendar", {"day": "tuesday", "bot_say": "hm tuesday"})) == "ok"
    assert seen == {"day": "tuesday"}
    assert asyncio.run(t.run_tool("nope", {})) == "Unknown tool nope."


def test_clip_decodes_html_entities():
    assert t._clip("Here&#39;s   what &lt;x&gt;") == "Here's what <x>"


def test_hand_off_task_returns_immediately_and_queues_ufo(monkeypatch):
    import sys
    import types

    sent = []

    async def send_task(text, call_id):
        await asyncio.sleep(0.2)  # UFO being slow must not delay the tool result
        sent.append((call_id, text))
        return "turn-1"

    monkeypatch.setitem(sys.modules, "post_call", types.SimpleNamespace(SPEAKER="Michael", MY_EMAIL="x@y", send_task=send_task))
    monkeypatch.setattr(t.gb, "is_configured", lambda: False)  # doesn't need GBrain

    async def go():
        import time
        t0 = time.perf_counter()
        out = await t.run_tool("hand_off_task", {"task": "email Michael a summary of this call", "bot_say": "yep"},
                               {"id": "c1", "customer": {"number": "+15555550100"}})
        fast = time.perf_counter() - t0 < 0.05
        await asyncio.sleep(0.3)
        return out, fast

    out, fast = asyncio.run(go())
    assert out.startswith("Queued") and fast
    assert sent and sent[0][0] == "c1"
    assert "email Michael a summary of this call" in sent[0][1]
