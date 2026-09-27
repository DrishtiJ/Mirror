# Mirror agent

The working voice agent behind Mirror: it answers your phone in your cloned voice, knows your live
context, and hands follow-up work to agents.

| Piece | What it does | Files |
|---|---|---|
| Phone (Vapi) | Answers calls in the cloned voice (Fish Audio), injects per-call context, exposes live tools, proxies River | `vapi/` |
| Context (GBrain) | Where you are (phone GPS), your next plan + drive time (calendar), who's calling (memory, email) | `situation.py`, `gbrain_context.py`, `location_server.py` |
| Tasks (UFO) | Mid-call and post-call follow-ups: notes to GBrain, email drafts (never sent) | `post_call.py`, `river_tools.py`, `ufo_ext_river/` |
| Model (River) | LoRA fine-tune of DeepSeek-V4.1-Flash on the speaker's talk transcripts for phrasing and cadence | `training/` |

## Run
```bash
cp .env.example .env                       # fill in keys; SPEAKER_NAME, DEMO_CONTACTS_JSON, ...
uv sync
uv run gbrain_context.py login             # connect GBrain (OAuth, once)
uv run location_server.py                  # :8790 phone GPS -> GBrain + tools API for UFO
uv run --with fastapi --with uvicorn --with httpx uvicorn vapi.fish_bridge:app --port 8788
ngrok http 8788                            # set VOICE_BRIDGE_URL, then: uv run --with httpx vapi/setup_assistant.py
```
UFO runs from [ufo-core](https://github.com/ufo-ai/ufo-core) with `ufo_ext_river` installed
(`uv pip install -e ufo_ext_river`, add `"river"` to the assistant pack) and `uv run --no-sync ufoctl serve`.

Tests: `uv run --with fastapi --with httpx pytest -q`
