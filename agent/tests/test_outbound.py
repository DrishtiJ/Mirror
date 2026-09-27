import asyncio

import pytest

import outbound


@pytest.fixture(autouse=True)
def env(monkeypatch, tmp_path):
    monkeypatch.setattr(outbound, "LEDGER", tmp_path / "calls.json")
    monkeypatch.setenv("DEMO_CONTACTS_JSON", '{"+15555550100": "Sam"}')
    monkeypatch.setenv("VAPI_ASSISTANT_ID", "asst")
    monkeypatch.setenv("VAPI_PHONE_NUMBER_ID", "pn")
    monkeypatch.setenv("VAPI_API_KEY", "k")
    monkeypatch.setattr(outbound, "SPEAKER", "Michael")  # tests pin the name; .env may set another


def test_kill_switch_off_by_default(monkeypatch):
    monkeypatch.delenv("OUTBOUND_CALLS_ENABLED", raising=False)
    with pytest.raises(outbound.CallRefused, match="off"):
        asyncio.run(outbound.place_call("+15555550100", "late"))


def test_only_known_contacts(monkeypatch):
    monkeypatch.setenv("OUTBOUND_CALLS_ENABLED", "1")
    with pytest.raises(outbound.CallRefused, match="not in DEMO_CONTACTS_JSON"):
        asyncio.run(outbound.place_call("+12125550199", "late"))


def test_dedupe_blocks_second_call(monkeypatch):
    monkeypatch.setenv("OUTBOUND_CALLS_ENABLED", "1")
    outbound._record("5555550100:late:e1", {"at": "earlier"})
    with pytest.raises(outbound.CallRefused, match="not calling twice"):
        asyncio.run(outbound.place_call("+15555550100", "late", dedupe_key="late:e1"))


def test_payload_carries_reason_and_situation():
    p = outbound.build_call("+15555550100", "Running 10 min late.", "Sam Lee", "You're driving.", None)
    ov = p["assistantOverrides"]
    assert p["customer"] == {"number": "+15555550100", "name": "Sam Lee"}
    assert ov["firstMessage"] == "Hey Sam, it's Michael."
    assert "Why you're calling: Running 10 min late." in ov["variableValues"]["notes"]
    assert "You're driving." in ov["variableValues"]["notes"]
    assert ov["metadata"]["direction"] == "outbound"


def test_outbound_number_preferred(monkeypatch):
    monkeypatch.setenv("VAPI_OUTBOUND_PHONE_NUMBER_ID", "twilio-pn")
    assert outbound.build_call("+15555550100", "r", "Sam", "n", None)["phoneNumberId"] == "twilio-pn"
    monkeypatch.delenv("VAPI_OUTBOUND_PHONE_NUMBER_ID")
    assert outbound.build_call("+15555550100", "r", "Sam", "n", None)["phoneNumberId"] == "pn"
