"""The server hands end-of-call reports to post_call in the background, once per call, with the caller name."""

import asyncio
import sys
import types

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("FISH_API_KEY", "x")
    monkeypatch.setenv("FISH_VOICE_ID", "y")
    handed = []

    async def hand_off_call(report):
        handed.append(report)
        return "turn-1"

    async def situation_for_caller(number):
        return {"caller_name": "Sam", "notes": "n"}

    monkeypatch.setitem(sys.modules, "post_call", types.SimpleNamespace(hand_off_call=hand_off_call))
    monkeypatch.setitem(sys.modules, "situation", types.SimpleNamespace(
        situation_for_caller=situation_for_caller, prewarm=lambda: None))
    from vapi import fish_bridge

    fish_bridge._HANDED_OFF.clear()
    fish_bridge._CALLER_NAMES.clear()
    with TestClient(fish_bridge.app) as c:
        c.handed = handed
        yield c


def post(c, type_, call_id="call-1"):
    return c.post("/vapi/assistant-request", json={"message": {
        "type": type_, "call": {"id": call_id, "customer": {"number": "+15555550100"}}}})


def test_report_is_handed_off_once_with_caller_name(client):
    assert post(client, "assistant-request").json()["assistantOverrides"]["variableValues"]["caller_name"] == "Sam"
    assert post(client, "end-of-call-report").json() == {}
    assert post(client, "end-of-call-report").json() == {}  # duplicate delivery
    for _ in range(20):
        if client.handed:
            break
        asyncio.run(asyncio.sleep(0.01))
    assert len(client.handed) == 1
    assert client.handed[0]["message"]["caller_name"] == "Sam"


def test_other_events_are_ignored(client):
    assert post(client, "status-update").json() == {}
    assert client.handed == []
