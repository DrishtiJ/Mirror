#!/usr/bin/env bash
# Starts the context + back-office side of the demo (idempotent):
#   location_server.py + tools API  :8790   (phone GPS -> GBrain; river_tools for UFO)
#   cloudflared quick tunnel -> :8790       (public URL for the phone app; changes on restart!)
#   UFO (ufoctl serve)              :8710   (must use --no-sync or uv drops the river extension)
# Vapi server (:8788) + ngrok are started separately (vapi/, other session).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LOGS="$ROOT/out/logs"; mkdir -p "$LOGS"

up() { curl -s -m 2 -o /dev/null "http://127.0.0.1:$1/"; }

for _ in 1 2 3 4 5; do pgrep -f "location_server.py" >/dev/null && ! up 8790 && sleep 1 || break; done  # let a restart finish
if ! up 8790; then
  (cd "$ROOT" && nohup uv run location_server.py >"$LOGS/location_server.log" 2>&1 </dev/null &)
  echo "started location_server (:8790)"
fi
if ! pgrep -f "cloudflared tunnel --url http://localhost:8790" >/dev/null; then
  nohup cloudflared tunnel --url http://localhost:8790 >"$LOGS/cloudflared.log" 2>&1 </dev/null &
  sleep 6
  echo "NEW tunnel (update the phone app): $(grep -o 'https://[a-z0-9-]*\.trycloudflare\.com' "$LOGS/cloudflared.log" | head -1)/location?token=<LOCATION_TOKEN>"
fi
if ! up 8710; then
  (cd "$HOME/Workspace/ufo-core" && nohup uv run --no-sync ufoctl serve >"$LOGS/ufo.log" 2>&1 </dev/null &)
  echo "started UFO (:8710)"
fi
sleep 5
for p in 8790 8710 8788; do up $p && echo "ok  :$p" || echo "DOWN :$p"; done
