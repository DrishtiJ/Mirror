#!/usr/bin/env bash
# Switch the live Vapi assistant's LLM:  vapi/switch_llm.sh river   |   vapi/switch_llm.sh openai
# river needs RIVER_BASE_URL / RIVER_API_KEY / RIVER_MODEL in .env. Restarts the proxy server, smoke-tests
# River through the tunnel first, and refuses to switch if River doesn't answer.
set -euo pipefail
cd "$(dirname "$0")/.."
target="${1:?river|openai}"
set_env() { grep -q "^$1=" .env && sed -i '' "s|^$1=.*|$1=$2|" .env || echo "$1=$2" >> .env; }
pkill -f "uvicorn vapi.fish_bridge" || true; sleep 1
nohup uv run --with fastapi --with uvicorn --with httpx uvicorn vapi.fish_bridge:app --port 8788 >> out/fish_bridge.log 2>&1 &
sleep 8
if [ "$target" = river ]; then
  tunnel=$(grep '^VOICE_BRIDGE_URL=' .env | cut -d= -f2 | sed 's|/vapi/tts||')
  echo "smoke-testing River through $tunnel/river ..."
  start=$(python3 -c 'import time; print(time.time())')
  out=$(curl -sN --max-time 20 -X POST "$tunnel/river/chat/completions" -H 'Content-Type: application/json' \
    -d '{"messages":[{"role":"system","content":"You are Michael on a phone call. Reply in one short sentence."},{"role":"user","content":"hey where you at?"}],"call":{"id":"x"},"metadata":{}}' | head -c 4000)
  echo "$out" | grep -q '"content"' || { echo "River did NOT answer; staying on current model:"; echo "$out" | head -c 500; exit 1; }
  python3 -c "import time; print(f'River full reply in {time.time()-$start:.2f}s')"
fi
set_env VAPI_LLM "$target"
uv run --with httpx vapi/setup_assistant.py | grep -E "model=|Vapi "
echo "switched to $target"
