"""Text-only end-to-end check of the Vapi assistant with live situation context.

Builds the same variableValues the phone path uses (situation_for_caller), then talks to the
real assistant through Vapi's Chat API, so the model, prompt and context are all the production ones.

    uv run scripts/chat_test.py +14155550123 "hey, where are you at?" "are we still on?"
"""

import asyncio
import os
import sys
from pathlib import Path

import httpx
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import situation  # noqa: E402

load_dotenv(Path(__file__).resolve().parent.parent / ".env")


async def main(number: str, turns: list[str]) -> None:
    ctx = await situation.situation_for_caller(number)
    values = {"speaker_name": "Michael", "caller_name": ctx["caller_name"] or "the caller", "notes": ctx["notes"]}
    print("--- context ---\n" + ctx["notes"] + "\n---------------")
    headers = {"Authorization": f"Bearer {os.environ['VAPI_API_KEY']}"}
    previous = None
    async with httpx.AsyncClient(timeout=60, headers=headers) as client:
        for text in turns:
            body = {"assistantId": os.environ["VAPI_ASSISTANT_ID"], "input": text,
                    "assistantOverrides": {"variableValues": values}}
            if previous:
                body["previousChatId"] = previous
            r = await client.post("https://api.vapi.ai/chat", json=body)
            if r.is_error:
                print(f"Vapi {r.status_code}: {r.text}")
                return
            data = r.json()
            previous = data.get("id")
            reply = " ".join(m.get("content", "") for m in data.get("output", []) if m.get("role") == "assistant")
            print(f"caller: {text}\nagent:  {reply}")


if __name__ == "__main__":
    args = sys.argv[1:] or ["+14155550123"]
    turns = args[1:] or ["Hey Michael, it's Sam. Where are you at?", "Cool, are we still on for coffee?",
                         "What should I tell Priya to bring?"]
    asyncio.run(main(args[0], turns))
