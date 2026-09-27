"""Tools API for back-office agents (UFO) acting for the phone agent's owner, backed by GBrain.

Mounted into location_server.py on 127.0.0.1:8790 under /tools. Every request needs
`Authorization: Bearer <RIVER_TOOLS_TOKEN>` from .env.

    GET  /tools/situation?caller=+1...   where the owner is, what's next, ETA, who's calling
    GET  /tools/calendar?day=YYYY-MM-DD  that day's events
    GET  /tools/memory?q=...             search GBrain memory
    GET  /tools/email?q=...              search Gmail (read only)
    POST /tools/remember  {"content", "slug"?}           save a note to GBrain memory
    POST /tools/draft_email {"to", "subject", "body"}    leave an email in Gmail Drafts (never sends)
    POST /tools/call {"to", "reason", "dedupe_key"?}     phone agent calls a demo contact (see outbound.py)
    GET  /tools/late_check               one line, "ok" or "LATE ~N min · event · start · id" (monitor probe)
    GET  /tools/late_status              the same as JSON (late, late_by_min, eta_min, event...)
    GET  /tools/contacts                 people the phone agent may call
"""

import json
import os
import secrets
from datetime import date

from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

import gbrain_context as gb
import outbound
import situation

def _authorized(request: Request) -> bool:
    token = os.getenv("RIVER_TOOLS_TOKEN", "")  # read per request: .env loads after import
    got = request.headers.get("authorization", "").removeprefix("Bearer ").strip()
    return bool(token) and secrets.compare_digest(got, token)


def _guard(handler):
    async def wrapped(request: Request) -> JSONResponse:
        if not _authorized(request):
            return JSONResponse({"error": "unauthorized"}, status_code=401)
        try:
            return await handler(request)
        except ValueError as e:
            return JSONResponse({"error": str(e)}, status_code=400)

    return wrapped


async def _gbrain(tool: str, args: dict) -> str:
    async with gb.gbrain_session() as session:
        result = await session.call_tool(tool, args)
    text = gb.result_text(result)
    if result.isError:
        raise ValueError(f"GBrain {tool} failed: {text}")
    return text


@_guard
async def get_situation(request: Request) -> JSONResponse:
    return JSONResponse(await situation.situation_for_caller(request.query_params.get("caller")))


@_guard
async def get_calendar(request: Request) -> JSONResponse:
    day = date.fromisoformat(request.query_params.get("day") or gb.local_now().date().isoformat())
    now = gb.local_now()
    events = await gb.schedule_for(now.replace(year=day.year, month=day.month, day=day.day))
    return JSONResponse({"day": day.isoformat(), "events": events})


@_guard
async def get_memory(request: Request) -> JSONResponse:
    q = request.query_params.get("q") or ""
    if not q:
        raise ValueError("q is required")
    return JSONResponse({"query": q, "results": await gb.search_memory(q)})


@_guard
async def get_email(request: Request) -> JSONResponse:
    q = request.query_params.get("q") or ""
    if not q:
        raise ValueError("q is required")
    return JSONResponse({"query": q, "results": await _gbrain("gmail_search", {"query": q, "limit": 10})})


@_guard
async def post_remember(request: Request) -> JSONResponse:
    body = await request.json()
    if not body.get("content"):
        raise ValueError("content is required")
    args = {"content": body["content"]}
    if body.get("slug"):
        args["slug"] = body["slug"]
    return JSONResponse({"saved": await _gbrain("capture", args)})


@_guard
async def post_draft_email(request: Request) -> JSONResponse:
    body = await request.json()
    missing = [k for k in ("to", "subject", "body") if not body.get(k)]
    if missing:
        raise ValueError(f"missing: {', '.join(missing)}")
    args = {k: body[k] for k in ("to", "subject", "body")}
    return JSONResponse({"draft": await _gbrain("gmail_draft", args)})


@_guard
async def post_call(request: Request) -> JSONResponse:
    body = await request.json()
    if not body.get("to") or not body.get("reason"):
        raise ValueError("to and reason are required")
    try:
        result = await outbound.place_call(body["to"], body["reason"], dedupe_key=body.get("dedupe_key"))
    except outbound.CallRefused as e:
        return JSONResponse({"refused": str(e)}, status_code=409)
    return JSONResponse(result)


@_guard
async def get_late_check(request: Request):
    from starlette.responses import PlainTextResponse

    snap = await situation.current_snapshot(timeout=5)
    snap.now = gb.local_now()
    return PlainTextResponse(situation.late_line(situation.late_status(snap)) + "\n")


@_guard
async def get_late_status(request: Request) -> JSONResponse:
    snap = await situation.current_snapshot(timeout=5)
    snap.now = gb.local_now()
    return JSONResponse({"status": situation.late_status(snap)})


@_guard
async def get_contacts(request: Request) -> JSONResponse:
    contacts = json.loads(os.getenv("DEMO_CONTACTS_JSON") or "{}")
    return JSONResponse({"contacts": [{"name": n, "number": num} for num, n in contacts.items()]})


@_guard
async def post_refresh(request: Request) -> JSONResponse:
    """Rebuild the situation snapshot now (location/calendar), e.g. right after a demo ping."""
    situation._snapshot = await situation.build_snapshot()
    situation._snapshot.now = gb.local_now()
    return JSONResponse({"late_status": situation.late_status(situation._snapshot)})


routes = [
    Route("/tools/situation", get_situation, methods=["GET"]),
    Route("/tools/calendar", get_calendar, methods=["GET"]),
    Route("/tools/memory", get_memory, methods=["GET"]),
    Route("/tools/email", get_email, methods=["GET"]),
    Route("/tools/remember", post_remember, methods=["POST"]),
    Route("/tools/draft_email", post_draft_email, methods=["POST"]),
    Route("/tools/call", post_call, methods=["POST"]),
    Route("/tools/late_check", get_late_check, methods=["GET"]),
    Route("/tools/late_status", get_late_status, methods=["GET"]),
    Route("/tools/contacts", get_contacts, methods=["GET"]),
    Route("/tools/refresh", post_refresh, methods=["POST"]),
]
