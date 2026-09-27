"""Create or update the Vapi assistant from .env.

    uv run --with httpx vapi/setup_assistant.py

.env:
  VAPI_API_KEY=...              (private key, dashboard.vapi.ai -> API Keys)
  VAPI_ASSISTANT_ID=...         (optional: update this assistant instead of creating one)
  VAPI_LLM=openai|river         (default openai)
  VAPI_OPENAI_MODEL=gpt-4.1-mini
  RIVER_BASE_URL / RIVER_API_KEY / RIVER_MODEL   (for VAPI_LLM=river)
  VAPI_VOICE=vapi|custom        (default vapi = Vapi's built-in low-latency voice)
  VAPI_VOICE_ID=Elliot          (built-in voice name; also the fallback for custom)
  VOICE_BRIDGE_URL=https://<ngrok>/vapi/tts     (for VAPI_VOICE=custom: fish_bridge.py or gradium_bridge.py)
"""

import json
import os
import re
import sys
from pathlib import Path

import httpx
from dotenv import load_dotenv, set_key

ROOT = Path(__file__).resolve().parent.parent
ENV = ROOT / ".env"
load_dotenv(ENV)

# Same prompt as the LiveKit agent; {x} placeholders become Liquid vars Vapi fills per call.
SPEAKER = os.getenv("SPEAKER_NAME", "Michael")
DEFAULTS = {"speaker_name": SPEAKER, "caller_name": "the caller", "notes": "No prior context for this call."}
prompt = re.sub(
    r"\{(\w+)\}",
    lambda m: '{{ %s | default: "%s" }}' % (m.group(1), DEFAULTS.get(m.group(1), "")),
    (ROOT / "system_prompt.txt").read_text(),
)


def model_cfg() -> dict:
    messages = [{"role": "system", "content": prompt}]
    tools = []
    if os.getenv("VOICE_BRIDGE_URL"):
        sys.path.insert(0, str(ROOT))
        from vapi.gbrain_tools import vapi_tools

        tools = vapi_tools(os.environ["VOICE_BRIDGE_URL"].replace("/vapi/tts", "/vapi/tool"))
    if os.getenv("VAPI_LLM") == "river":
        return {
            "provider": "custom-llm",
            # Through our proxy (fish_bridge /river): strips Vapi's extra fields, pins model, no thinking.
            "url": os.environ["VOICE_BRIDGE_URL"].replace("/vapi/tts", "/river"),  # Vapi appends /chat/completions
            "model": os.environ["RIVER_MODEL"],
            "messages": messages,
            "tools": tools,
            "maxTokens": 90,
            "temperature": 0.6,
        }
    return {
        "provider": "openai",
        "model": os.getenv("VAPI_OPENAI_MODEL", "gpt-4.1-mini"),
        "messages": messages,
        "tools": tools,
        "maxTokens": 90,
        "temperature": 0.6,
    }


def voice_cfg() -> dict:
    builtin = {"provider": "vapi", "voiceId": os.getenv("VAPI_VOICE_ID", "Elliot")}
    # custom = any bridge (fish_bridge.py or gradium_bridge.py) exposed at VOICE_BRIDGE_URL
    if os.getenv("VAPI_VOICE") in ("custom", "fish", "gradium"):
        url = os.getenv("VOICE_BRIDGE_URL") or os.environ["GRADIUM_BRIDGE_URL"]
        return {
            "provider": "custom-voice",
            "server": {"url": url, "timeoutSeconds": 20},
            "fallbackPlan": {"voices": [builtin]},
        }
    return builtin


assistant = {
    "name": "River voice agent",
    "firstMessage": "Hey, it's {{ speaker_name | default: \"%s\" }}. What's up?" % SPEAKER,
    "firstMessageMode": "assistant-speaks-first",
    "model": model_cfg(),
    "voice": voice_cfg(),
    "transcriber": {"provider": "deepgram", "model": "nova-3", "language": "en", "endpointing": 150},
    # Latency: the smart endpointer averaged ~1.2 s of waiting per turn. Use short
    # transcript-based endpointing instead: reply fast after punctuation, a bit later without.
    "startSpeakingPlan": {
        "waitSeconds": 0.1,
        "smartEndpointingEnabled": False,
        "transcriptionEndpointingPlan": {
            "onPunctuationSeconds": 0.3,
            "onNoPunctuationSeconds": 0.8,
            "onNumberSeconds": 0.4,
        },
    },
    # Only the events we use; everything else is noise on the tunnel.
    "serverMessages": ["end-of-call-report"],
    # Web/dashboard calls don't pass through the phone number's server, so the assistant points
    # at the same endpoint for its end-of-call-report (duplicates are dropped by call id).
    **({"server": {"url": os.environ["VOICE_BRIDGE_URL"].replace("/vapi/tts", "/vapi/assistant-request")}}
       if os.getenv("VOICE_BRIDGE_URL") else {}),
    "stopSpeakingPlan": {"numWords": 0, "voiceSeconds": 0.2, "backoffSeconds": 1},
}

headers = {"Authorization": f"Bearer {os.environ['VAPI_API_KEY']}"}
aid = os.getenv("VAPI_ASSISTANT_ID")
with httpx.Client(base_url="https://api.vapi.ai", headers=headers, timeout=30) as c:
    r = c.patch(f"/assistant/{aid}", json=assistant) if aid else c.post("/assistant", json=assistant)
    if r.is_error:
        sys.exit(f"Vapi {r.status_code}: {r.text}")
    body = r.json()

if not aid:
    set_key(str(ENV), "VAPI_ASSISTANT_ID", body["id"])
print(json.dumps({k: body.get(k) for k in ("id", "name")}, indent=2))
print(f"model={assistant['model']['provider']} voice={assistant['voice']['provider']}")
print(f"Test: dashboard.vapi.ai -> Assistants -> {body['id']} -> Talk to Assistant, or open vapi/call.html")
