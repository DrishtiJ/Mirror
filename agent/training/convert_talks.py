"""Talk transcripts -> short phone-style SFT pairs that carry the speaker's phrasing and cadence.

Input: {video_id: {"title", "lang", "text"}} (YouTube transcripts). Talks are monologues, and
training on them raw teaches the agent to lecture. So each talk is cut into short spoken bits
(1-3 sentences, ~15-45 words, fillers kept), and the base model writes the brief thing a caller
might have said right before each bit. Every pair is wrapped in the production system prompt.

    uv run training/convert_talks.py ~/Downloads/trainging.txt --max 400   # -> training/out/style_sft.jsonl
"""

import argparse
import json
import os
import random
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")

import river_client as river  # noqa: E402

from call_context import DEFAULT_CONTEXT, CallContext  # noqa: E402

MODEL = os.getenv("RIVER_BASE_MODEL", "deepseek-ai/DeepSeek-V4.1-Flash")
OUT = Path(__file__).resolve().parent / "out" / "style_sft.jsonl"
MIN_WORDS, MAX_WORDS = 15, 45
# Talk-specific openers/closers that would be odd on a phone call
SKIP = re.compile(r"\b(slide|this talk|today's (talk|presentation)|thank you all|welcome to|q&a|questions\?)\b", re.I)

PROMPT = """Here is something a person said out loud, casually, as a reply on a phone call:

"{bit}"

Write the one short thing the other person on the call most likely said right before it, so it reads as a
natural reply. Casual, spoken, at most 15 words. Output only that line, no quotes."""


def bits_from(text: str) -> list[str]:
    """Split into sentences (or ~20-word runs when ASR has no punctuation) and pack into 15-45 word bits."""
    text = re.sub(r"\s+", " ", text.replace("\n", " ")).strip().lstrip("- ")
    sentences = re.split(r"(?<=[.!?])\s+", text)
    if len(sentences) < len(text.split()) / 60:  # unpunctuated ASR: cut at soft boundaries
        words, sentences, cur = text.split(), [], []
        for w in words:
            cur.append(w)
            if len(cur) >= 20 and w.lower() in {"so", "and", "but", "right", "okay", "yes", "now"}:
                sentences.append(" ".join(cur[:-1]))
                cur = [w]
        sentences.append(" ".join(cur))
    bits, cur = [], []
    for s in sentences:
        cur.append(s)
        n = len(" ".join(cur).split())
        if n >= MIN_WORDS:
            if n <= MAX_WORDS:
                bits.append(" ".join(cur))
            cur = []
    return [b for b in bits if not SKIP.search(b)]


def caller_line(client: river.Client, bit: str) -> str | None:
    r = client.chat_complete([{"role": "user", "content": PROMPT.format(bit=bit)}], base_model=MODEL,
                             max_tokens=40, temperature=0.8, chat_template_kwargs={"thinking": False})
    line = (json.loads(r.response_json)["choices"][0]["message"].get("content") or "").strip().strip('"')
    return line if 0 < len(line.split()) <= 20 else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("talks", type=Path)
    ap.add_argument("--max", type=int, default=400)
    args = ap.parse_args()
    talks = json.loads(args.talks.expanduser().read_text())
    bits = [b for t in talks.values() if str(t.get("lang", "en")).startswith("en") for b in bits_from(t["text"])]
    rng = random.Random(0)
    rng.shuffle(bits)
    bits = bits[: args.max]
    print(f"{len(bits)} bits from {sum(1 for t in talks.values() if str(t.get('lang','en')).startswith('en'))} English talks")
    client = river.Client(api_key=os.environ["RIVER_API_KEY"])

    def one(bit: str):
        try:
            line = caller_line(client, bit)
        except Exception as e:
            print("skip:", repr(e)[:100])
            return None
        if not line:
            return None
        system = CallContext(speaker_name=DEFAULT_CONTEXT.speaker_name, caller_name="the caller",
                             notes=DEFAULT_CONTEXT.notes).render_prompt()
        return {"messages": [{"role": "system", "content": system}, {"role": "user", "content": line},
                             {"role": "assistant", "content": bit}]}

    with ThreadPoolExecutor(max_workers=8) as pool:
        rows = [r for r in pool.map(one, bits) if r]
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text("".join(json.dumps(r) + "\n" for r in rows))
    print(f"kept {len(rows)} -> {OUT}")
    for r in rows[:3]:
        print(f"  caller: {r['messages'][1]['content']}\n  reply:  {r['messages'][2]['content']}")


if __name__ == "__main__":
    main()
