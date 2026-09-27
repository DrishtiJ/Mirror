"""River vs gpt-4.1 on a real phone-agent turn: same system prompt + live context, short reply.

River (no dedicated deployment yet) goes through the queued chat_complete API: no streaming, so
this measures total reply time, not time-to-first-token. gpt-4.1 goes through LiveKit Inference
(streaming, so we get both).

    uv run scripts/bench_llm.py [runs]
"""

import asyncio
import json
import os
import statistics
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")

import river_client as river  # noqa: E402

import situation  # noqa: E402
from call_context import CallContext  # noqa: E402

RIVER_MODELS = os.getenv("BENCH_RIVER_MODELS", "Qwen/Qwen3.6-35B-A3B-FP8").split(",")
USER_TURN = "Hey Michael, it's Sam. Where are you at right now? Are we still on?"


def pct(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(p / 100 * (len(xs) - 1))))]


async def messages() -> list[dict]:
    situation.prewarm()
    await situation._first_build
    ctx = await situation.situation_for_caller("+15555550100")
    call = CallContext(speaker_name="Michael", caller_name=ctx["caller_name"] or "the caller", notes=ctx["notes"])
    return [{"role": "system", "content": call.render_prompt()}, {"role": "user", "content": USER_TURN}]


def bench_river(msgs, model, runs):
    c = river.Client(api_key=os.environ["RIVER_API_KEY"])
    times, replies = [], []
    for _ in range(runs):
        t0 = time.perf_counter()
        r = c.chat_complete(msgs, base_model=model, max_tokens=90, temperature=0.6,
                            chat_template_kwargs={"enable_thinking": False})
        times.append(time.perf_counter() - t0)
        body = json.loads(r.response_json)
        replies.append(body["choices"][0]["message"].get("content", "").strip())
    return times, replies


async def bench_gpt41(msgs, runs):
    from livekit.agents import inference
    from livekit.agents.llm import ChatContext

    llm = inference.LLM(model="openai/gpt-4.1")
    ttfts, totals, replies = [], [], []
    for _ in range(runs):
        ctx = ChatContext.empty()
        for m in msgs:
            ctx.add_message(role=m["role"], content=m["content"])
        t0, first, text = time.perf_counter(), None, ""
        async with llm.chat(chat_ctx=ctx) as stream:
            async for chunk in stream:
                if chunk.delta and chunk.delta.content:
                    first = first or time.perf_counter()
                    text += chunk.delta.content
        totals.append(time.perf_counter() - t0)
        ttfts.append((first or time.perf_counter()) - t0)
        replies.append(text.strip())
    return ttfts, totals, replies


def summary(name, xs):
    return f"{name}: median {statistics.median(xs):.2f}s  p90 {pct(xs, 90):.2f}s  (n={len(xs)})"


async def main():
    runs = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    msgs = await messages()
    for model in RIVER_MODELS:
        times, replies = await asyncio.to_thread(bench_river, msgs, model, runs)
        print(summary(f"River {model} total (queued, no streaming)", times))
        print("  e.g.:", replies[0])
    ttfts, totals, replies = await bench_gpt41(msgs, runs)
    print(summary("gpt-4.1 time-to-first-token", ttfts))
    print(summary("gpt-4.1 total", totals))
    print("  e.g.:", replies[0])


asyncio.run(main())
