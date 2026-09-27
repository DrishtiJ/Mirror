"""On-stage trigger for the late-watcher demo: make Michael "late", and have UFO check now.

    uv run scripts/demo_late.py            # phone "moves" ~an hour out, UFO runs the late check now
    uv run scripts/demo_late.py --reset    # also forget earlier late calls, so it can call again
    uv run scripts/demo_late.py --near     # put the phone back near the venue (on time)

What happens: a simulated GPS ping goes through location_server (-> GBrain), the situation
snapshot is rebuilt, and UFO's late-watch conversation gets "run the late check now". UFO then
decides on its own (river_late_status / river_situation) and places the call via river_call.
The UFO scheduled task would catch it within 2 minutes anyway; this just makes it immediate.
"""

import json
import os
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

import httpx
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

FAR = (37.4636, -122.4286, 27.0)   # Half Moon Bay, ~45-60 min drive from Oakland, driving
NEAR = (37.7680, -122.2155, 27.0)  # I-880 near 47th Ave, ~7 min from Blue Bottle Webster St
BASE = "http://127.0.0.1:8790"
TOOLS = {"Authorization": f"Bearer {os.environ['RIVER_TOOLS_TOKEN']}"}


def ping(lat: float, lon: float, speed: float) -> None:
    body = {"locations": [{"type": "Feature", "geometry": {"type": "Point", "coordinates": [lon, lat]},
                           "properties": {"timestamp": datetime.now(timezone.utc).isoformat(),
                                          "speed": speed, "motion": ["driving"]}}]}
    r = httpx.post(f"{BASE}/location", params={"token": os.environ["LOCATION_TOKEN"]}, json=body, timeout=10)
    r.raise_for_status()


def main() -> None:
    args = set(sys.argv[1:])
    if "--reset" in args:
        ledger = ROOT / "out" / "outbound_calls.json"
        if ledger.exists():
            calls = {k: v for k, v in json.loads(ledger.read_text()).items() if ":late:" not in k}
            ledger.write_text(json.dumps(calls, indent=2))
        print("forgot earlier late calls")
    lat, lon, speed = NEAR if "--near" in args else FAR
    ping(lat, lon, speed)
    # location_server writes the first fix to GBrain right away; later ones every ~20 s
    import time

    time.sleep(float(os.getenv("DEMO_GBRAIN_WAIT_S", "22")))
    status = httpx.post(f"{BASE}/tools/refresh", headers=TOOLS, timeout=30).json()["late_status"]
    print("late_status:", status)
    if "--near" in args:
        return
    token = (Path.home() / ".ufoctl" / "token").read_text().strip()
    r = httpx.post("http://127.0.0.1:8710/surface/ufo/late-watch", timeout=10,
                   headers={"authorization": f"Bearer {token}", "x-ufo-send": "1", "x-ufo-send-id": str(uuid.uuid4())},
                   content=b"Run the late check now (same steps as the scheduled late-watch task).")
    print("UFO:", r.text.strip())


if __name__ == "__main__":
    main()
