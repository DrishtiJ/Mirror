"""Serve a trained checkpoint on a River dedicated deployment and smoke-test it.

    uv run training/deploy_river.py training/out/train_run_dj-cadence-1pass.json

Prints the OpenAI-compatible base_url for Vapi (RIVER_BASE_URL) and a streamed sample reply
with time-to-first-token. If deployment creation is refused (gated), falls back to one queued
sample from the checkpoint so the fine-tune can still be shown.
"""

import json
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

import river_client as river  # noqa: E402

MSGS = [{"role": "system", "content": "You are Michael, talking with Sam on a live phone call. Short, spoken turns."},
        {"role": "user", "content": "hey, where you at? we still on?"}]


def main() -> None:
    run = json.loads(Path(sys.argv[1]).read_text())
    ckpt, base = run["inference_checkpoint"], run["base_model"]
    client = river.Client(api_key=os.environ["RIVER_API_KEY"])
    print("checkpoint:", ckpt)
    t0 = time.time()
    try:
        dep = client.create_deployment(checkpoint=ckpt, unified_replicas=1, idempotency_key=run["name"],
                                       wait=True, wait_timeout=900)
    except Exception as e:
        print(f"deployment refused/failed after {time.time() - t0:.0f}s: {e}")
        r = client.chat_complete_from_checkpoint(MSGS, checkpoint_path=ckpt, base_model=base, max_tokens=80,
                                                 chat_template_kwargs={"thinking": False})
        print("queued sample from checkpoint:", json.loads(r.response_json)["choices"][0]["message"].get("content"))
        return
    print(f"deployment {dep.id} ready in {time.time() - t0:.0f}s\nRIVER_BASE_URL={dep.base_url}")
    from openai import OpenAI

    oai = OpenAI(api_key=os.environ["RIVER_API_KEY"], base_url=dep.base_url, max_retries=3)
    t1, first, text = time.time(), None, ""
    for chunk in oai.chat.completions.create(model=base, messages=MSGS, stream=True, max_tokens=80,
                                             extra_body={"chat_template_kwargs": {"thinking": False}}):
        if chunk.choices and chunk.choices[0].delta.content:
            first = first or time.time()
            text += chunk.choices[0].delta.content
    print(f"TTFT {first - t1:.2f}s, total {time.time() - t1:.2f}s: {text}")


if __name__ == "__main__":
    main()
