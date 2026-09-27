"""Times UFO on the inbound demo path: a mid-call task, then the end-of-call report, for one fake call.
Creates a real Gmail draft (to SPEAKER_EMAIL) and a GBrain note.

    uv run scripts/ufo_speed_test.py
"""

import asyncio
import sys
import time
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import post_call  # noqa: E402

CALL_ID = f"speedtest-{uuid.uuid4().hex[:8]}"


async def main() -> None:
    t0 = time.perf_counter()
    mid = (f"Live phone call {CALL_ID} with Sam (+15555550100) is still in progress. During the call, Michael agreed "
           "to this task: email Priya the demo link and the pitch deck. "
           "Do it now with your river_* tools (drafts only, never send). The end-of-call report for this same call "
           "will arrive later; don't do this task twice. Only this call's own conversation counts as already done: "
           "drafts or notes from other calls don't, so do the task. Be quick: at most one lookup; if you don't have a "
           "recipient's address, draft it to owner@example.com with subject '[For <name>] ...'.")
    print("mid-call turn:", await post_call.send_task(mid, CALL_ID))
    res = await post_call.read_result(CALL_ID)
    print(f"[{time.perf_counter() - t0:5.1f}s] mid-call result:", res)
    report = {"message": {"type": "end-of-call-report", "endedReason": "customer-ended-call", "durationSeconds": 95,
              "startedAt": "2026-09-27T22:30:00Z", "caller_name": "Sam",
              "call": {"id": CALL_ID, "customer": {"number": "+15555550100"}},
              "analysis": {"summary": "Sam confirmed coffee at 4 with Priya; Michael will email Priya the demo link "
                                      "and the pitch deck."},
              "artifact": {"transcript": "User: Hey, still on for 4?\nAI: Yep, see you at Blue Bottle.\n"
                                         "User: Priya wants to see the demo before coffee.\n"
                                         "AI: I'll email her a list right now."}}}
    t1 = time.perf_counter()
    print("end-of-call turn:", await post_call.hand_off_call(report))
    res = await post_call.read_result(CALL_ID)
    print(f"[{time.perf_counter() - t1:5.1f}s] end-of-call result:", res)
    print(f"total {time.perf_counter() - t0:.1f}s")


asyncio.run(main())
