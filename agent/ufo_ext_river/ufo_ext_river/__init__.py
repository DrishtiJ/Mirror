"""UFO extension: tools that act for the phone agent's owner, via the river-voice-agent tools API.

The API (river_tools.py) runs on RIVER_TOOLS_URL (default http://127.0.0.1:8790) and holds the
GBrain credentials; this extension only carries UFO_RIVER_TOOLS_TOKEN.
"""

import os

import httpx
from pydantic import BaseModel, Field

from ufo.sdk.manifest import Manifest
from ufo.sdk.tools import TextContent, ToolContext, ToolDef, ToolResult

NAME = "river"
VERSION = "0.1.0"
TIMEOUT_S = 30


async def _request(method: str, path: str, **kwargs) -> ToolResult:
    base = os.getenv("UFO_RIVER_TOOLS_URL", "http://127.0.0.1:8790")
    headers = {"authorization": f"Bearer {os.getenv('UFO_RIVER_TOOLS_TOKEN', '')}"}
    async with httpx.AsyncClient(timeout=TIMEOUT_S, headers=headers) as client:
        r = await client.request(method, base + path, **kwargs)
    return ToolResult(content=(TextContent(text=f"HTTP {r.status_code}\n{r.text}"),))


class SituationInput(BaseModel):
    caller: str | None = Field(None, description="Optional caller phone number in E.164, e.g. +14155550123")


class CalendarInput(BaseModel):
    day: str = Field(description="Date as YYYY-MM-DD")


class QueryInput(BaseModel):
    q: str = Field(description="What to search for")


class RememberInput(BaseModel):
    content: str = Field(description="Markdown to save to the owner's GBrain memory")
    slug: str | None = Field(None, description="Optional page slug, e.g. people/priya or calls/2026-09-27-sam")


class DraftInput(BaseModel):
    to: str = Field(description="Recipient email address")
    subject: str
    body: str = Field(description="Plain-text email body, written and signed as the owner (the task names them)")


class CallInput(BaseModel):
    to: str = Field(description="Phone number in E.164, e.g. +15555550100 (must be one of the owner's demo contacts)")
    reason: str = Field(description="Why the owner is calling, in one or two sentences, e.g. 'Running ~10 min late to "
                        "coffee at Blue Bottle; still on.'")
    dedupe_key: str | None = Field(None, description="Same key = never call twice, e.g. 'late:<event id>'")


async def call(ctx: ToolContext, args: CallInput) -> ToolResult:
    return await _request("POST", "/tools/call", json=args.model_dump(exclude_none=True))


class NoInput(BaseModel):
    pass


async def contacts(ctx: ToolContext, args: NoInput) -> ToolResult:
    return await _request("GET", "/tools/contacts")


async def late_status(ctx: ToolContext, args: NoInput) -> ToolResult:
    return await _request("GET", "/tools/late_status")


async def situation(ctx: ToolContext, args: SituationInput) -> ToolResult:
    return await _request("GET", "/tools/situation", params={"caller": args.caller} if args.caller else None)


async def calendar(ctx: ToolContext, args: CalendarInput) -> ToolResult:
    return await _request("GET", "/tools/calendar", params={"day": args.day})


async def memory(ctx: ToolContext, args: QueryInput) -> ToolResult:
    return await _request("GET", "/tools/memory", params={"q": args.q})


async def email(ctx: ToolContext, args: QueryInput) -> ToolResult:
    return await _request("GET", "/tools/email", params={"q": args.q})


async def remember(ctx: ToolContext, args: RememberInput) -> ToolResult:
    return await _request("POST", "/tools/remember", json=args.model_dump(exclude_none=True))


async def draft_email(ctx: ToolContext, args: DraftInput) -> ToolResult:
    return await _request("POST", "/tools/draft_email", json=args.model_dump())


def _tool(name: str, description: str, model, handler) -> ToolDef:
    return ToolDef(name=name, description=description, input_model=model, handler=handler,
                   binds_member_authority=False)


def manifest() -> Manifest:
    return Manifest(
        name=NAME,
        version=VERSION,
        tools=(
            _tool("river_situation", "The owner's live situation: where their phone is, how they're moving, their next "
                  "calendar plan with drive time, and who a caller is (pass their number).", SituationInput, situation),
            _tool("river_calendar", "the owner's Google Calendar events for one day (read only).", CalendarInput, calendar),
            _tool("river_memory_search", "Search the owner's GBrain memory: people, companies, past calls, notes.",
                  QueryInput, memory),
            _tool("river_email_search", "Search the owner's Gmail (read only), same syntax as Gmail's search box.",
                  QueryInput, email),
            _tool("river_remember", "Save a note to the owner's GBrain memory, e.g. a call summary or a follow-up.",
                  RememberInput, remember),
            _tool("river_draft_email", "Write an email as the owner, signed with their name, and leave it in Gmail Drafts for them to "
                  "review. It is never sent automatically.", DraftInput, draft_email),
            _tool("river_call", "Have the owner's phone agent CALL someone now, in the owner's voice, and talk it "
                  "through with them. A real phone rings: only call when it clearly helps (e.g. they're running late "
                  "and they're waiting). Refused unless the number is a known contact; one call per dedupe_key.",
                  CallInput, call),
            _tool("river_contacts", "People the owner's phone agent is allowed to call, with numbers.", NoInput, contacts),
            _tool("river_late_status", "Is the owner running late to their next event? Drive time from their phone's "
                  "location vs the event start: late, late_by_min, eta_min, event, start, event_id.", NoInput,
                  late_status),
        ),
    )
