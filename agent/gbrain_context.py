"""Personal context from GBrain (gbrain.io): where the speaker is, their schedule, and what
the brain knows about the caller. It's merged into the prompt's {notes} slot per call.

GBrain exposes one MCP server (https://gbrain.io/mcp) behind OAuth. Sign in once:

    uv run gbrain_context.py login      # opens the browser, stores tokens in .gbrain/
    uv run gbrain_context.py tools      # prints GBrain's real tool names + input schemas
    uv run gbrain_context.py preview [caller]  # prints the context block the agent would get

GBrain doesn't document its MCP tool names, so tools are picked by name pattern and
arguments by schema. After `tools`, pin the exact names with GBRAIN_TOOL_MEMORY,
GBRAIN_TOOL_CALENDAR and GBRAIN_TOOL_EMAIL if the patterns guess wrong.

Everything here is best-effort: a slow or failing GBrain never blocks or breaks a call.
"""

import asyncio
import json
import logging
import os
import re
import sys
import webbrowser
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from threading import Thread
from typing import Any
from urllib.parse import parse_qs, urlparse
from zoneinfo import ZoneInfo

import httpx
from mcp import ClientSession
from mcp.client.auth import OAuthClientProvider
from mcp.client.streamable_http import streamablehttp_client
from mcp.shared.auth import OAuthClientInformationFull, OAuthClientMetadata, OAuthToken

logger = logging.getLogger("gbrain")

HERE = Path(__file__).parent
GBRAIN_URL = os.getenv("GBRAIN_MCP_URL", "https://gbrain.io/mcp")
TOKEN_DIR = HERE / ".gbrain"
CALLBACK_PORT = 8765
REDIRECT_URI = f"http://localhost:{CALLBACK_PORT}/callback"

# Pre-call fetch budget. The greeting waits at most this long for GBrain.
FETCH_TIMEOUT_S = float(os.getenv("GBRAIN_TIMEOUT_S", "3.0"))
# Location comes only from phone pings stored in GBrain, one per line:
#   LOCATION UPDATE 2026-09-27T14:05:00-07:00 | YC office, 1 Main St, San Francisco | 37.79,-122.39
# A ping older than this is treated as unknown rather than guessed.
LOCATION_TAG = "LOCATION UPDATE"
LOCATION_MAX_AGE_H = float(os.getenv("GBRAIN_LOCATION_MAX_AGE_H", "6"))
LOCATION_RE = re.compile(re.escape(LOCATION_TAG) + r"\s+(\S+)\s*\|\s*([^|\n]+?)\s*(?:\|[^\n]*)?$", re.MULTILINE)

# Default name patterns for GBrain's tools (first match wins; override with env).
TOOL_PATTERNS = {
    "location": (os.getenv("GBRAIN_TOOL_LOCATION"), r"location"),
    "memory": (os.getenv("GBRAIN_TOOL_MEMORY"), r"(memory|brain|recall).*(search|query|recall)|^(search|recall)$"),
    "calendar": (os.getenv("GBRAIN_TOOL_CALENDAR"), r"(gcal|calendar).*(list|events|search)"),
    "email": (os.getenv("GBRAIN_TOOL_EMAIL"), r"(gmail|mail).*(search|list)"),
}
QUERY_ARGS = ("query", "q", "text", "search", "question", "prompt")
LIMIT_ARGS = ("limit", "max_results", "maxResults", "top_k", "k", "count")
TIME_MIN_ARGS = ("time_min", "timeMin", "start", "from", "after", "start_time")
TIME_MAX_ARGS = ("time_max", "timeMax", "end", "to", "before", "end_time")


# --- OAuth ------------------------------------------------------------------------------


class FileTokenStorage:
    """Stores GBrain OAuth tokens and the registered client in .gbrain/ (gitignored)."""

    def __init__(self, directory: Path | None = None) -> None:
        self.dir = directory or TOKEN_DIR  # resolved at call time, so TOKEN_DIR can be switched

    def _read(self, name: str) -> dict | None:
        path = self.dir / name
        return json.loads(path.read_text()) if path.exists() else None

    def _write(self, name: str, data: str) -> None:
        self.dir.mkdir(mode=0o700, exist_ok=True)
        path = self.dir / name
        path.write_text(data)
        path.chmod(0o600)

    async def get_tokens(self) -> OAuthToken | None:
        data = self._read("tokens.json")
        return OAuthToken.model_validate(data) if data else None

    async def set_tokens(self, tokens: OAuthToken) -> None:
        self._write("tokens.json", tokens.model_dump_json())

    async def get_client_info(self) -> OAuthClientInformationFull | None:
        data = self._read("client.json")
        return OAuthClientInformationFull.model_validate(data) if data else None

    async def set_client_info(self, client_info: OAuthClientInformationFull) -> None:
        self._write("client.json", client_info.model_dump_json())


def is_configured() -> bool:
    return os.getenv("GBRAIN_ENABLED", "1") != "0" and (TOKEN_DIR / "tokens.json").exists()


async def _no_interactive_login(url: str) -> None:
    raise RuntimeError("GBrain needs a sign-in: run `uv run gbrain_context.py login`")


async def _wait_for_callback() -> tuple[str, str | None]:
    """One-shot localhost server that catches the OAuth redirect."""
    result: dict[str, str | None] = {}

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            params = parse_qs(urlparse(self.path).query)
            result["code"] = params.get("code", [None])[0]
            result["state"] = params.get("state", [None])[0]
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"GBrain connected. You can close this tab.")

        def log_message(self, *_: Any) -> None:
            pass

    server = HTTPServer(("localhost", CALLBACK_PORT), Handler)
    thread = Thread(target=server.handle_request, daemon=True)
    thread.start()
    await asyncio.to_thread(thread.join, 300)
    server.server_close()
    if not result.get("code"):
        raise RuntimeError("GBrain sign-in didn't complete")
    return result["code"], result.get("state")


def _auth(interactive: bool) -> OAuthClientProvider:
    async def open_browser(url: str) -> None:
        print(f"Opening GBrain sign-in: {url}")
        webbrowser.open(url)

    return OAuthClientProvider(
        server_url=GBRAIN_URL,
        client_metadata=OAuthClientMetadata(
            client_name="River voice agent",
            redirect_uris=[REDIRECT_URI],
            grant_types=["authorization_code", "refresh_token"],
            response_types=["code"],
            token_endpoint_auth_method="none",
            # Least privilege: read calendar/mail/schedule; memory write only for location pings.
            scope=os.getenv("GBRAIN_SCOPES", "memory:full calendar:read gmail:read schedule:read"),
        ),
        storage=FileTokenStorage(),
        redirect_handler=open_browser if interactive else _no_interactive_login,
        callback_handler=_wait_for_callback if interactive else None,
    )


TOKEN_URL = os.getenv("GBRAIN_TOKEN_URL", "https://gbrain.io/oauth/token")
REFRESH_MARGIN_S = 300  # refresh this long before the access token expires
_refresh_lock = asyncio.Lock()


def _token_age_s() -> float:
    return datetime.now().timestamp() - (TOKEN_DIR / "tokens.json").stat().st_mtime


async def refresh_tokens(force: bool = False) -> bool:
    """Swap the refresh token for a new access token when it is (nearly) expired.

    GBrain answers an expired access token with 403 token_expired, and the MCP SDK only refreshes
    on 401, so long-running processes do it themselves. tokens.json's mtime is the issue time."""
    path = TOKEN_DIR / "tokens.json"
    if not path.exists():
        return False
    async with _refresh_lock:
        tokens = json.loads(path.read_text())
        expires_in = float(tokens.get("expires_in") or 3600)
        if not force and _token_age_s() < expires_in - REFRESH_MARGIN_S:
            return False
        client = json.loads((TOKEN_DIR / "client.json").read_text())
        data = {"grant_type": "refresh_token", "refresh_token": tokens["refresh_token"],
                "client_id": client["client_id"], "resource": GBRAIN_URL}
        if client.get("client_secret") and client.get("token_endpoint_auth_method") == "client_secret_post":
            data["client_secret"] = client["client_secret"]
        async with httpx.AsyncClient(timeout=15) as http:
            r = await http.post(TOKEN_URL, data=data)
        if r.is_error:
            logger.error("GBrain token refresh failed: %s %s", r.status_code, r.text)
            return False
        new = r.json()
        new.setdefault("refresh_token", tokens["refresh_token"])  # keep it if not rotated
        FileTokenStorage()._write("tokens.json", json.dumps(new))
        logger.info("GBrain access token refreshed (expires in %ss)", new.get("expires_in"))
        return True


@asynccontextmanager
async def gbrain_session(*, interactive: bool = False):
    if not interactive:
        await refresh_tokens()
    async with streamablehttp_client(GBRAIN_URL, auth=_auth(interactive), timeout=10) as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()
            yield session


# --- Tool selection and calls -----------------------------------------------------------


def pick_tool(tools: list[Any], kind: str) -> Any | None:
    pinned, pattern = TOOL_PATTERNS[kind]
    for tool in tools:
        if pinned and tool.name == pinned:
            return tool
    if pinned:
        logger.warning("GBRAIN_TOOL_%s=%s not found on the server", kind.upper(), pinned)
        return None
    return next((t for t in tools if re.search(pattern, t.name, re.IGNORECASE)), None)


def build_args(tool: Any, *, query: str | None = None, limit: int | None = None,
               time_min: datetime | None = None, time_max: datetime | None = None) -> dict:
    """Fill whichever of our inputs the tool's JSON schema has a slot for."""
    props = (tool.inputSchema or {}).get("properties", {})

    def first(names: tuple[str, ...]) -> str | None:
        return next((n for n in names if n in props), None)

    args: dict[str, Any] = {}
    for names, value in (
        (QUERY_ARGS, query),
        (LIMIT_ARGS, limit),
        (TIME_MIN_ARGS, time_min.isoformat() if time_min else None),
        (TIME_MAX_ARGS, time_max.isoformat() if time_max else None),
    ):
        key = first(names)
        if key and value is not None:
            args[key] = value
    return args


def result_text(result: Any) -> str:
    parts = []
    for item in result.content or []:
        text = getattr(item, "text", None)
        if text:
            parts.append(text)
    if not parts and getattr(result, "structuredContent", None):
        parts.append(json.dumps(result.structuredContent))
    return "\n".join(parts).strip()


async def call(session: ClientSession, tools: list[Any], kind: str, **kwargs: Any) -> str:
    tool = pick_tool(tools, kind)
    if tool is None:
        return ""
    args = build_args(tool, **kwargs)
    try:
        result = await session.call_tool(tool.name, args)
    except Exception as e:
        logger.warning("GBrain %s (%s) failed: %s", kind, tool.name, e)
        return ""
    if result.isError:
        logger.warning("GBrain %s (%s) returned an error: %s", kind, tool.name, result_text(result))
        return ""
    return result_text(result)


# --- Pre-call context -------------------------------------------------------------------


@dataclass
class PersonalContext:
    now: datetime
    location: str = ""
    schedule: str = ""
    caller_notes: str = ""
    caller_emails: str = ""
    sources: list[str] = field(default_factory=list)

    def render(self) -> str:
        """Plain-text lines for the prompt's {notes} slot."""
        lines = [f"Right now: {self.now:%A %B %-d, %Y, %-I:%M %p} {self.now.tzname()}."]
        if self.location:
            lines.append(f"Where you are: {self.location}")
        if self.schedule:
            lines.append(f"Your schedule today and tomorrow:\n{self.schedule}")
        if self.caller_notes:
            lines.append(f"What you know about them:\n{self.caller_notes}")
        if self.caller_emails:
            lines.append(f"Recent emails with them:\n{self.caller_emails}")
        return "\n".join(lines)


def local_now() -> datetime:
    return datetime.now(ZoneInfo(os.getenv("AGENT_TIMEZONE", "America/Los_Angeles")))


def latest_location(text: str, now: datetime) -> str:
    """Newest well-formed phone ping in `text`, if it's fresh enough. Never guesses."""
    best: tuple[datetime, str] | None = None
    for stamp, place in LOCATION_RE.findall(text):
        try:
            at = datetime.fromisoformat(stamp)
        except ValueError:
            continue
        if at.tzinfo is None:
            at = at.replace(tzinfo=now.tzinfo)
        if best is None or at > best[0]:
            best = (at, place.strip())
    if best is None:
        return ""
    age = now - best[0]
    if age > timedelta(hours=LOCATION_MAX_AGE_H) or age < -timedelta(minutes=5):
        logger.info("latest location ping is %s old; treating location as unknown", age)
        return ""
    minutes = int(age.total_seconds() // 60)
    when = "just now" if minutes < 2 else f"{minutes} minutes ago" if minutes < 90 else f"{minutes // 60} hours ago"
    return f"{best[1]} (from your phone, {when})"


async def fetch_location(session: ClientSession, tools: list[Any], now: datetime) -> str:
    """A dedicated GBrain location tool if one exists, else the tagged phone pings in memory."""
    if pick_tool(tools, "location") is not None:
        raw = await call(session, tools, "location")
        return latest_location(raw, now) or raw
    return latest_location(await call(session, tools, "memory", query=LOCATION_TAG, limit=10), now)


def merge_notes(existing: str, ctx: PersonalContext, *, default: str = "") -> str:
    """Caller-supplied notes first, then GBrain's lines. Drops the placeholder default."""
    parts = [existing.strip()] if existing.strip() and existing.strip() != default.strip() else []
    parts.append(ctx.render())
    return "\n".join(parts)


def _known_caller(caller_name: str) -> bool:
    return bool(caller_name.strip()) and caller_name.strip().lower() not in ("the caller", "caller", "unknown")


async def fetch_personal_context(speaker_name: str, caller_name: str) -> PersonalContext:
    """Pull location, schedule and caller context from GBrain in parallel."""
    now = local_now()
    ctx = PersonalContext(now=now, location=os.getenv("AGENT_LOCATION", ""))
    if not is_configured():
        return ctx

    day_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
    async with gbrain_session() as session:
        tools = (await session.list_tools()).tools

        async def nothing() -> str:
            return ""

        known = _known_caller(caller_name)
        location, schedule, notes, emails = await asyncio.gather(
            fetch_location(session, tools, now),
            call(session, tools, "calendar", time_min=day_start, time_max=day_start + timedelta(days=2), limit=100),
            call(session, tools, "memory", query=caller_name, limit=10) if known else nothing(),
            call(session, tools, "email", query=caller_name, limit=25) if known else nothing(),
        )
    ctx.location = location or ctx.location  # a fresh phone ping beats the AGENT_LOCATION fallback
    ctx.schedule, ctx.caller_notes, ctx.caller_emails = schedule, notes, emails
    ctx.sources = [k for k, v in (("location", location), ("calendar", schedule),
                                  ("memory", notes), ("email", emails)) if v]
    return ctx


async def fetch_personal_context_safe(speaker_name: str, caller_name: str) -> PersonalContext:
    """fetch_personal_context under FETCH_TIMEOUT_S; any failure degrades to time + env location."""
    try:
        ctx = await asyncio.wait_for(fetch_personal_context(speaker_name, caller_name), FETCH_TIMEOUT_S)
        logger.info("GBrain context loaded from: %s", ", ".join(ctx.sources) or "nothing")
        return ctx
    except Exception as e:  # includes TimeoutError and auth failures
        logger.warning("GBrain context unavailable, continuing without it: %r", e)
        return PersonalContext(now=local_now(), location=os.getenv("AGENT_LOCATION", ""))


# --- Mid-call lookups (used by PersonaAgent tools) ---------------------------------------


async def search_memory(query: str) -> str:
    async with gbrain_session() as session:
        tools = (await session.list_tools()).tools
        return await call(session, tools, "memory", query=query, limit=10)


async def schedule_for(day: datetime) -> str:
    start = day.replace(hour=0, minute=0, second=0, microsecond=0)
    async with gbrain_session() as session:
        tools = (await session.list_tools()).tools
        return await call(session, tools, "calendar", time_min=start, time_max=start + timedelta(days=1), limit=100)


# --- CLI --------------------------------------------------------------------------------


async def _main(cmd: str) -> None:
    from dotenv import load_dotenv

    load_dotenv(HERE / ".env")
    if cmd == "login":
        async with gbrain_session(interactive=True) as session:
            tools = (await session.list_tools()).tools
        print(f"Signed in. GBrain exposes {len(tools)} tools; tokens saved to {TOKEN_DIR}/")
    elif cmd == "tools":
        async with gbrain_session() as session:
            tools = (await session.list_tools()).tools
        for t in tools:
            print(f"- {t.name}: {(t.description or '').splitlines()[0] if t.description else ''}")
            print(f"    args: {json.dumps((t.inputSchema or {}).get("properties", {}))}")
        for kind in TOOL_PATTERNS:
            chosen = pick_tool(tools, kind)
            print(f"{kind:>8} -> {chosen.name if chosen else 'NO MATCH (set GBRAIN_TOOL_' + kind.upper() + ')'}")
    elif cmd == "preview":
        caller = sys.argv[2] if len(sys.argv) > 2 else "the caller"
        ctx = await fetch_personal_context(os.getenv("SPEAKER_NAME", "Michael"), caller)
        print(ctx.render())
    else:
        print(__doc__)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(_main(sys.argv[1] if len(sys.argv) > 1 else "help"))
