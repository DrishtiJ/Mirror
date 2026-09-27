"""Vapi custom-voice webhook -> Fish Audio TTS (your cloned voice).

    uv run --with fastapi --with uvicorn --with httpx uvicorn vapi.fish_bridge:app --port 8788
    ngrok http 8788      # then VOICE_BRIDGE_URL=https://<ngrok>/vapi/tts, VAPI_VOICE=custom, rerun setup_assistant.py

.env: FISH_API_KEY, FISH_VOICE_ID (the cloned voice's model id from fish.audio),
      FISH_MODEL (default s2.1-pro), FISH_LATENCY (default low: low | balanced | normal).
Vapi posts one sentence per request and wants raw 16-bit mono PCM at message.sampleRate.
"""

import asyncio
import json
import logging
import os
import sys
import time
from pathlib import Path

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")
sys.path.insert(0, str(ROOT))  # so the optional situation.py at the project root imports
log = logging.getLogger("uvicorn.error")

FISH_URL = "https://api.fish.audio/v1/tts"
HEADERS = {
    "Authorization": f"Bearer {os.environ['FISH_API_KEY']}",
    "model": os.getenv("FISH_MODEL", "s2.1-pro"),
}
VOICE_ID = os.environ["FISH_VOICE_ID"]
LATENCY = os.getenv("FISH_LATENCY", "low")

app = FastAPI()
client = httpx.AsyncClient(timeout=httpx.Timeout(20, connect=5), http2=False)


@app.post("/vapi/tts")
async def vapi_tts(request: Request) -> StreamingResponse:
    message = (await request.json())["message"]
    text, rate = message["text"], message["sampleRate"]
    body = {"text": text, "reference_id": VOICE_ID, "format": "pcm", "sample_rate": rate, "latency": LATENCY}
    t0 = time.perf_counter()

    async def pcm():
        first = True
        async with client.stream("POST", FISH_URL, headers=HEADERS, json=body) as r:
            if r.is_error:
                log.error("fish %s: %s", r.status_code, (await r.aread())[:500])
                r.raise_for_status()
            async for chunk in r.aiter_bytes():
                if first:
                    log.info("fish ttfb %.0f ms for %r", 1000 * (time.perf_counter() - t0), text[:60])
                    first = False
                yield chunk

    return StreamingResponse(pcm(), media_type="application/octet-stream")


@app.on_event("startup")
async def prewarm_situation() -> None:
    """Start situation.py's background refresher so calls only wait on the caller lookup."""
    try:
        import situation

        situation.prewarm()
        log.info("situation prewarm started")
    except Exception:
        log.exception("situation prewarm failed; calls will build context inline")


ASSISTANT_ID = os.getenv("VAPI_ASSISTANT_ID")
_CALLER_NAMES: dict[str, str] = {}  # call id -> resolved caller name, for the post-call handoff
_HANDED_OFF: set[str] = set()      # call ids already sent to UFO (reports can arrive twice)
_BACKGROUND: set[asyncio.Task] = set()


def _hand_off_to_ufo(body: dict) -> None:
    """Fire-and-forget: post_call.hand_off_call queues the follow-up in UFO and never raises."""
    message = body.setdefault("message", {})
    call_id = (message.get("call") or {}).get("id") or ""
    if call_id in _HANDED_OFF:
        return
    _HANDED_OFF.add(call_id)
    if call_id in _CALLER_NAMES:
        message["caller_name"] = _CALLER_NAMES.pop(call_id)
    try:
        import post_call  # other workstream, project root
    except Exception:
        log.exception("post_call unavailable; skipping UFO handoff for %s", call_id)
        return

    async def run() -> None:
        turn = await post_call.hand_off_call(body)
        log.info("UFO handoff for call %s -> turn %s", call_id, turn)

    task = asyncio.create_task(run())
    _BACKGROUND.add(task)
    task.add_done_callback(_BACKGROUND.discard)
SITUATION_TIMEOUT_S = 2.5  # Vapi allows 7.5 s end-to-end for the whole assistant-request


@app.post("/vapi/assistant-request")
async def assistant_request(request: Request) -> dict:
    """Inbound call: pick the assistant and fill per-call context from the caller's number.

    Context comes from situation.situation_for_caller(number) (owned by another workstream);
    any failure or timeout falls back to the assistant's default Liquid values.
    """
    body = await request.json()
    message = body.get("message", {})
    # The phone number's server URL receives EVERY call event (status, transcripts, reports...).
    # assistant-request needs an answer; end-of-call-report kicks off the UFO follow-up in the
    # background; everything else is acknowledged instantly.
    if message.get("type") == "end-of-call-report":
        _hand_off_to_ufo(body)
        return {}
    if message.get("type") != "assistant-request":
        return {}
    number = ((message.get("call") or {}).get("customer") or {}).get("number")
    variables: dict[str, str] = {"speaker_name": os.getenv("SPEAKER_NAME", "Michael")}
    t0 = time.perf_counter()
    try:
        import situation  # lazy: optional module at the project root

        found = await asyncio.wait_for(situation.situation_for_caller(number), SITUATION_TIMEOUT_S)
        variables |= {k: v for k, v in (found or {}).items() if k in ("caller_name", "notes") and v}
    except Exception:
        log.exception("situation lookup failed for %s; using defaults", number)
    call_id = (message.get("call") or {}).get("id")
    if call_id and variables.get("caller_name"):
        _CALLER_NAMES[call_id] = variables["caller_name"]
    log.info("assistant-request from %s -> %s (%.0f ms)", number, variables, 1000 * (time.perf_counter() - t0))
    return {"assistantId": ASSISTANT_ID, "assistantOverrides": {"variableValues": variables}}


@app.post("/vapi/tool")
async def tool_calls(request: Request) -> dict:
    """Vapi tool-calls -> GBrain lookups (vapi/gbrain_tools.py). All calls in one message run in parallel."""
    from vapi import gbrain_tools

    message = (await request.json()).get("message", {})
    calls = message.get("toolCallList") or []

    async def one(call: dict) -> dict:
        fn = call.get("function") or {}
        args = fn.get("arguments") or {}
        if isinstance(args, str):
            args = json.loads(args or "{}")
        t0 = time.perf_counter()
        result = await gbrain_tools.run_tool(fn.get("name", ""), args, message.get("call"))
        log.info("tool %s(%s) %.0f ms -> %r", fn.get("name"), args, 1000 * (time.perf_counter() - t0), result[:120])
        return {"toolCallId": call.get("id"), "result": result}

    return {"results": await asyncio.gather(*(one(c) for c in calls))}


OPENAI_FIELDS = {"model", "messages", "tools", "tool_choice", "temperature", "max_tokens", "top_p",
                 "stop", "stream", "parallel_tool_calls", "response_format", "seed"}


@app.post("/river/chat/completions")
async def river_proxy(request: Request) -> StreamingResponse:
    """Vapi custom-llm -> River deployment. Drops Vapi's extra fields (call, metadata, ...), pins the
    model, turns off thinking, and streams River's SSE straight back."""
    import httpx

    body = {k: v for k, v in (await request.json()).items() if k in OPENAI_FIELDS}
    body |= {"model": os.environ["RIVER_MODEL"], "stream": True,
             "chat_template_kwargs": {"thinking": False}}
    body.setdefault("max_tokens", 90)
    url = os.environ["RIVER_BASE_URL"].rstrip("/") + "/chat/completions"
    headers = {"Authorization": f"Bearer {os.environ['RIVER_API_KEY'].strip()}"}
    t0 = time.perf_counter()

    async def sse():
        first = True
        async with httpx.AsyncClient(timeout=httpx.Timeout(30, connect=5)) as client:
            async with client.stream("POST", url, headers=headers, json=body) as r:
                if r.is_error:
                    log.error("river %s: %s", r.status_code, (await r.aread())[:500])
                    r.raise_for_status()
                async for chunk in r.aiter_raw():
                    if first:
                        log.info("river ttfb %.0f ms", 1000 * (time.perf_counter() - t0))
                        first = False
                    yield chunk

    return StreamingResponse(sse(), media_type="text/event-stream")


@app.get("/health")
async def health() -> dict:
    return {"ok": True}
