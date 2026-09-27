"""Does the model say its own lead-in AND call the tool in the same reply? (gpt-4.1-mini via LiveKit Inference)"""
import asyncio, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from dotenv import load_dotenv
load_dotenv(ROOT / ".env")
from livekit.agents import inference, llm
from livekit.agents.llm import function_tool
from livekit.agents.utils import http_context
from call_context import CallContext
from vapi.gbrain_tools import SCHEMAS

NOTES = "It's Sunday, 2:40 PM. You're driving on 47th Avenue, Oakland. Next: coffee with Sam and Priya at 4 PM at Blue Bottle."
ASKS = [
    "Can you draft an email to me summarizing this conversation?",
    "Send Priya the deck before the meeting, would you?",
    "Hey are you free Tuesday afternoon?",
    "Did Priya ever send you her current policy?",
    "What do you know about Priya's company?",
    "Do I owe you anything from last week?",
    "What's on for tomorrow?",
]

def stub(schema):
    async def _impl(raw_arguments: dict):
        return "stub"
    return function_tool(_impl, raw_schema=schema)

async def main():
    http_context._new_session_ctx()
    m = inference.LLM(model=__import__("os").getenv("CHECK_MODEL", "openai/gpt-4.1-mini"))
    tools = [stub(s) for s in SCHEMAS]
    prompt = CallContext("Michael", "Sam", NOTES).render_prompt()
    for ask in ASKS:
        ctx = llm.ChatContext.empty(); ctx.add_message(role="system", content=prompt); ctx.add_message(role="user", content=ask)
        text, calls = "", []
        async with m.chat(chat_ctx=ctx, tools=tools) as s:
            async for c in s:
                if c.delta:
                    text += c.delta.content or ""
                    calls += [f"{t.name}({t.arguments})" for t in c.delta.tool_calls]
        import json as _j
        says = [_j.loads(c.split("(", 1)[1][:-1]).get("bot_say", "") for c in calls]
        ok = bool(calls) and all(x.strip() for x in says)
        print(f"{'OK ' if ok else 'NO '}| {ask}\n     bot_say: {says}  text: {text.strip()!r}\n     tools: {calls}")
    await http_context._close_http_ctx()

asyncio.run(main())
