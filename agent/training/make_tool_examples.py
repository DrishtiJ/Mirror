"""Tool-use SFT examples, distilled from the base model, so a style fine-tune keeps tool calling.

A LoRA trained only on plain transcripts can erode tool calls the live agent relies on
(check_calendar, lookup_person, recall, search_email, open_loops, hand_off_task, each with the
spoken `bot_say` stall). This generates examples with the *base* DeepSeek-V4.1-Flash as teacher:

  caller turn + realistic situation notes -> base model's tool call (kept only if bot_say is filled)
  -> a plausible tool result -> base model's short spoken answer

plus "no tool" turns where it should just talk. Output rows carry `tools` so training renders the
same tool block the live agent sends.

    uv run training/make_tool_examples.py            # -> training/out/tool_sft.jsonl
"""

import json
import os
import random
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "vapi")]
load_dotenv(ROOT / ".env")

import river_client as river  # noqa: E402

import gbrain_tools  # noqa: E402  (the live agent's tool schemas)
import situation as st  # noqa: E402
from call_context import CallContext  # noqa: E402

MODEL = os.getenv("RIVER_BASE_MODEL", "deepseek-ai/DeepSeek-V4.1-Flash")
OUT = Path(__file__).resolve().parent / "out" / "tool_sft.jsonl"
TZ = ZoneInfo("America/Los_Angeles")
TOOLS = [{"type": "function", "function": s} for s in gbrain_tools.SCHEMAS]


def notes_variants() -> list[tuple[str, str]]:
    """(caller_name, notes) situations rendered by the production renderer."""
    now = datetime(2026, 9, 28, 11, 40, tzinfo=TZ)
    coffee = st.Event(id="e1", title="Coffee with Dana and Priya", start=now + timedelta(minutes=50), end=None,
                      location="Blue Bottle Coffee, 300 Webster St, Oakland",
                      description="Dana is introducing Priya, who is building a robotics startup.")
    driving = st.Snapshot(now=now, fix=st.Fix(at=now, lat=0, lon=0, speed_mps=26, motion="driving", place="I-880, Oakland"),
                          next_event=coffee, events=[coffee], eta=(12, 6))
    standup = st.Event(id="e2", title="Team standup", start=now + timedelta(hours=3), end=None, location="Office")
    office = st.Snapshot(now=now, fix=st.Fix(at=now, lat=0, lon=0, speed_mps=0, place="Market St, San Francisco"),
                         next_event=standup, events=[standup])
    late = st.Snapshot(now=now, fix=st.Fix(at=now, lat=0, lon=0, speed_mps=12, motion="driving", place="Bay Bridge"),
                       next_event=coffee, events=[coffee], eta=(62, 14))
    evening = st.Snapshot(now=now.replace(hour=20, minute=15))
    return [
        ("Dana", st.render(driving, "Dana", st.meeting_fact("Dana", driving))),
        ("Sam", st.render(office, "Sam", [])),
        ("Priya", st.render(late, "Priya", st.meeting_fact("Priya", late))),
        ("the caller", st.render(evening, None, [])),
    ]


# (caller turn, expected tool or None)
SCENARIOS = [
    ("are you free tuesday afternoon?", "check_calendar"),
    ("what's your thursday look like?", "check_calendar"),
    ("can we do lunch tomorrow?", "check_calendar"),
    ("are you around friday morning for a call?", "check_calendar"),
    ("do you have anything tonight?", "check_calendar"),
    ("what do you know about Priya's company?", "lookup_person"),
    ("have you met Jordan from Acme before?", "lookup_person"),
    ("remind me what Dana does again?", "lookup_person"),
    ("what did we talk about last time?", "recall"),
    ("remember that restaurant you mentioned?", "recall"),
    ("what was the number you quoted me last week?", "recall"),
    ("did I send you the demo video?", "search_email"),
    ("did Priya ever email you her slides?", "search_email"),
    ("did you get my intro email to Jordan?", "search_email"),
    ("did the judging schedule come through?", "search_email"),
    ("what am I still waiting on from you?", "open_loops"),
    ("did you ever send that thing you promised?", "open_loops"),
    ("anything you owe me this week?", "open_loops"),
    ("can you email Priya the demo link?", "hand_off_task"),
    ("can you send me the pitch deck after this?", "hand_off_task"),
    ("could you intro me to Jordan over email?", "hand_off_task"),
    ("can we push to 4:30? can you update the invite?", "hand_off_task"),
    ("can you follow up with Dana about the intro?", "hand_off_task"),
    ("send me a recap of this call?", "hand_off_task"),
    ("hey, where are you at?", None),
    ("are we still on?", None),
    ("cool, see you soon", None),
    ("haha yeah totally", None),
    ("how's your day going?", None),
    ("sorry, can you hear me? bad signal", None),
    ("ok thanks, talk later", None),
    ("wait, who is this?", None),
]

FAKE_RESULTS = {
    "check_calendar": "Tuesday: 10:00 AM team sync; afternoon free after 1 PM. 5:30 PM gym.",
    "lookup_person": "Priya Nair. Building a robotics startup, team of 6, based in SF. Met through Dana at the YC hackathon.",
    "recall": "Last call (Sep 20): talked about her hackathon project and the demo order; you mentioned Pizzaiolo on Telegraph.",
    "search_email": "No email from Priya with attachments yet. Latest: Sep 24 'Re: quick intro' from Dana, cc Priya.",
    "open_loops": "You owe: send Priya the demo link (promised Sep 24). They owe: their slides.",
    "hand_off_task": "Queued. It'll be handled in the background; just confirm briefly and move on.",
}


def complete(client: river.Client, messages: list[dict]) -> dict:
    r = client.chat_complete(messages, base_model=MODEL, max_tokens=160, temperature=0.7, tools=TOOLS,
                             chat_template_kwargs={"thinking": False})
    return json.loads(r.response_json)["choices"][0]["message"]


def make(client: river.Client, caller: str, notes: str, turn: str, expected: str | None) -> dict | None:
    system = CallContext(speaker_name="Michael", caller_name=caller, notes=notes).render_prompt()
    msgs = [{"role": "system", "content": system}, {"role": "user", "content": turn}]
    first = complete(client, msgs)
    calls = first.get("tool_calls") or []
    if expected is None:
        if calls or not (first.get("content") or "").strip():
            return None
        return {"messages": msgs + [{"role": "assistant", "content": first["content"].strip()}], "tools": gbrain_tools.SCHEMAS}
    if len(calls) != 1 or calls[0]["function"]["name"] != expected:
        return None
    args = json.loads(calls[0]["function"]["arguments"])
    if not str(args.get("bot_say", "")).strip():
        return None
    call_id = calls[0].get("id") or "call_0"
    assistant = {"role": "assistant", "content": "", "tool_calls": [
        {"type": "function", "id": call_id, "function": {"name": expected, "arguments": json.dumps(args)}}]}
    tool = {"role": "tool", "tool_call_id": call_id, "name": expected, "content": FAKE_RESULTS[expected]}
    second = complete(client, msgs + [assistant, tool])
    answer = (second.get("content") or "").strip()
    if second.get("tool_calls") or not answer or len(answer.split()) > 45:
        return None
    return {"messages": msgs + [assistant, tool, {"role": "assistant", "content": answer}], "tools": gbrain_tools.SCHEMAS}


def main() -> None:
    client = river.Client(api_key=os.environ["RIVER_API_KEY"])
    variants = notes_variants()
    rng = random.Random(0)
    jobs = [(caller, notes, turn, exp) for turn, exp in SCENARIOS for caller, notes in rng.sample(variants, 2)]
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(lambda j: _safe(client, *j), jobs))
    kept = [r for r in results if r]
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text("".join(json.dumps(r) + "\n" for r in kept))
    tools_used = [r["messages"][2]["tool_calls"][0]["function"]["name"] for r in kept if r["messages"][2].get("tool_calls")]
    print(f"kept {len(kept)}/{len(jobs)} -> {OUT}  (tool examples {len(tools_used)}, no-tool {len(kept) - len(tools_used)})")


def _safe(client, *job):
    try:
        return make(client, *job)
    except Exception as e:
        print("skip:", job[2], repr(e)[:120])
        return None


if __name__ == "__main__":
    main()
