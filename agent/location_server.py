"""Receives phone GPS pings and files the latest fix in GBrain (page `location/current`).

    uv run location_server.py                         # listens on :8790
    cloudflared tunnel --url http://localhost:8790    # public URL for the phone

Phone side, either:
- Overland (iOS/Android GPS logger, free): Receiver Endpoint =
    https://<tunnel>/location?token=<LOCATION_TOKEN>
  It posts batches with speed and motion (driving/walking), which the agent speaks from.
- iOS Shortcuts: "Get Current Location" -> "Get Contents of URL" POST JSON
    {"lat": ..., "lon": ..., "speed_mps": ...}  to the same URL.

LOCATION_TOKEN (in .env) must match, so nobody else can move you around.
"""

import asyncio
import json
import logging
import os
import secrets
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

import uvicorn
from dotenv import load_dotenv
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

import gbrain_context as gb
import river_tools
import situation
from situation import LOCATION_SLUG

HERE = Path(__file__).parent
load_dotenv(HERE / ".env")
logger = logging.getLogger("location_server")

TOKEN = os.getenv("LOCATION_TOKEN", "")
PORT = int(os.getenv("LOCATION_PORT", "8790"))
# GBrain write at most this often; the newest fix always wins when it does write.
MIN_WRITE_INTERVAL_S = float(os.getenv("LOCATION_WRITE_INTERVAL_S", "20"))

_latest: dict | None = None
_last_written_at: str | None = None
_write_lock = asyncio.Lock()


def fixes_from_payload(body: dict) -> list[dict]:
    """Overland GeoJSON batches or a flat {lat, lon, ...} ping -> normalized fixes."""
    fixes = []
    for feat in body.get("locations") or []:
        coords = (feat.get("geometry") or {}).get("coordinates") or []
        props = feat.get("properties") or {}
        if len(coords) < 2:
            continue
        motion = props.get("motion") or []
        fixes.append({
            "at": props.get("timestamp") or datetime.now(timezone.utc).isoformat(),
            "lat": coords[1], "lon": coords[0],
            "speed_mps": props.get("speed"),
            "motion": motion[0] if isinstance(motion, list) and motion else (motion or ""),
        })
    if "lat" in body and "lon" in body:
        fixes.append({
            "at": body.get("at") or datetime.now(timezone.utc).isoformat(),
            "lat": float(body["lat"]), "lon": float(body["lon"]),
            "speed_mps": body.get("speed_mps"), "motion": body.get("motion") or "",
        })
    return sorted(fixes, key=lambda f: f["at"])


def page_content(fix: dict) -> str:
    return (
        "---\ntype: note\ntitle: Current location\n---\n"
        "# Current location\n\n"
        "Latest GPS fix from Michael's phone, written by location_server.py.\n\n"
        f"```json\n{json.dumps(fix)}\n```\n"
    )


async def write_to_gbrain() -> None:
    global _last_written_at
    async with _write_lock:
        fix = _latest
        if fix is None or fix["at"] == _last_written_at:
            return
        async with gb.gbrain_session() as session:
            result = await session.call_tool("put_page", {"slug": LOCATION_SLUG, "content": page_content(fix)})
        if result.isError:
            logger.error("GBrain put_page failed: %s", gb.result_text(result))
            return
        _last_written_at = fix["at"]
        logger.info("location filed in GBrain: %s", fix)


async def _writer_loop() -> None:
    while True:
        try:
            await write_to_gbrain()
        except Exception as e:
            logger.error("GBrain write failed: %r", e)
        await asyncio.sleep(MIN_WRITE_INTERVAL_S)


async def receive(request: Request) -> JSONResponse:
    global _latest
    if not TOKEN or not secrets.compare_digest(request.query_params.get("token", ""), TOKEN):
        return JSONResponse({"error": "bad token"}, status_code=401)
    fixes = fixes_from_payload(await request.json())
    if fixes:
        _latest = fixes[-1]
        if _last_written_at is None:
            asyncio.create_task(write_to_gbrain())  # first fix goes out right away
    # Overland wants {"result": "ok"} or it re-sends the batch
    return JSONResponse({"result": "ok", "received": len(fixes)})


async def latest(request: Request) -> JSONResponse:
    if not TOKEN or not secrets.compare_digest(request.query_params.get("token", ""), TOKEN):
        return JSONResponse({"error": "bad token"}, status_code=401)
    return JSONResponse({"latest": _latest, "written_to_gbrain": _last_written_at})


@asynccontextmanager
async def lifespan(app: Starlette):
    if not TOKEN:
        logger.error("LOCATION_TOKEN is not set in .env; every ping will be rejected")
    writer = asyncio.create_task(_writer_loop())
    situation.prewarm()
    yield
    writer.cancel()


app = Starlette(
    routes=[Route("/location", receive, methods=["POST"]), Route("/location", latest, methods=["GET"]),
            *river_tools.routes],
    lifespan=lifespan,
)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    uvicorn.run(app, host="127.0.0.1", port=PORT)
