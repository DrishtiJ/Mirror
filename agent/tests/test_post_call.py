import pytest

import post_call


@pytest.fixture(autouse=True)
def speaker(monkeypatch):
    monkeypatch.setattr(post_call, "SPEAKER", "Michael")  # tests pin the name; .env may set another


def test_build_task_from_vapi_report():
    report = {"message": {
        "type": "end-of-call-report", "endedReason": "customer-ended-call", "durationSeconds": 74.2,
        "startedAt": "2026-09-27T21:50:00Z", "caller_name": "Sam",
        "call": {"customer": {"number": "+15555550100"}},
        "analysis": {"summary": "Confirmed coffee at 4."},
        "artifact": {"transcript": "User: where are you?\nAI: ten minutes out."},
    }}
    task = post_call.build_task(report)
    assert "Sam (+15555550100) called Michael" in task
    assert "Duration: 74 s" in task and "customer-ended-call" in task
    assert "Confirmed coffee at 4." in task and "ten minutes out." in task
    assert "slug calls/2026-09-27-sam" in task  # no call id in this report
    assert "Never send" in task and "visible results first" in task


def test_build_task_unknown_caller_minimal_report():
    task = post_call.build_task({"message": {"call": {"customer": {"number": "+12125550199"}}}})
    assert "+12125550199 called Michael" in task and "slug calls/call-0199" in task


def test_build_task_outbound_call():
    report = {"message": {"call": {"type": "outboundPhoneCall", "customer": {"number": "+15555550100"},
              "assistantOverrides": {"metadata": {"direction": "outbound", "caller_name": "Sam",
                                                  "reason": "Running 10 min late."}}}}}
    task = post_call.build_task(report)
    assert "Michael's AI voice agent called Sam (+15555550100) as Michael to say: Running 10 min late." in task


def test_channel_per_call():
    assert post_call.channel_for("01A0E4E8-a20b-7000") == "call-01a0e4e8-a20b-7000"
    assert post_call.channel_for(None) == post_call.UFO_CHANNEL
    assert post_call.channel_for("x/../y") == "call-xy"


def test_note_slug_is_unique_per_call():
    task = post_call.build_task({"message": {"startedAt": "2026-09-27T22:00:00Z", "caller_name": "Sam",
                                             "call": {"id": "01a0e4e8-a20b", "customer": {"number": "+15555550100"}}}})
    assert "slug calls/2026-09-27-sam-01a0e4e8" in task
