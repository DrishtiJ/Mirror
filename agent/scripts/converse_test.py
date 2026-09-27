"""Simulated phone call, text only: live situation context -> the real system prompt ->
gpt-4.1-mini (the model the Vapi assistant runs), via LiveKit Inference.

    uv run scripts/converse_test.py +14155550123
"""

import asyncio
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")

from livekit.agents import Agent, AgentSession, inference  # noqa: E402

import situation  # noqa: E402
from call_context import CallContext  # noqa: E402

TURNS = [
    "Hey Michael, it's Sam. Where are you at right now?",
    "Oh, how far out is he?",
    "Nice. Are we still on for coffee?",
    "Cool. Priya wants to see the demo, can you send her the link?",
    "Wait, are you an AI?",
]


async def main(number: str) -> None:
    situation.prewarm()  # like the server does at startup, so the call sees the warm snapshot
    await situation._first_build
    ctx = await situation.situation_for_caller(number)
    call = CallContext(speaker_name="Michael", caller_name=ctx["caller_name"] or "the caller", notes=ctx["notes"])
    print("--- notes ---\n" + ctx["notes"] + "\n-------------")
    async with AgentSession(llm=inference.LLM(model="openai/gpt-4.1-mini")) as session:
        await session.start(Agent(instructions=call.render_prompt()))
        for text in TURNS:
            result = await session.run(user_input=text)
            reply = " ".join(e.item.text_content or "" for e in result.events if getattr(e, "type", "") == "message")
            print(f"caller: {text}\nagent:  {reply}\n")


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1] if len(sys.argv) > 1 else "+14155550123"))
