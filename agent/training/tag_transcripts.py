"""Recordings -> delivery-tagged River SFT data.

Transcribes each recording with Fish `transcribe-1-pro` (speaker markers + inline emotion /
vocal-event cues + timed segments), keeps only cues Fish TTS can render, adds [break] /
[long-break] where Michael actually paused, and writes multi-turn chat examples where
Michael's turns are the assistant turns River learns to write.

    uv run --with httpx training/tag_transcripts.py RECORDINGS_DIR --me-speaker 0
    uv run --with httpx training/tag_transcripts.py RECORDINGS_DIR --me-speaker auto   # the speaker who talks most

Outputs (training/out/):
  sft.jsonl       {"messages": [system, user, assistant, ...]} per recording
  cadence.json    reply gap, words/min, turn length, filler rate, tag counts (for Vapi timing)
  asr_cache/      raw Fish responses, so reruns don't re-bill
Needs FISH_API_KEY with API credit (transcription has no free tier).
"""

import argparse
import json
import os
import re
import statistics
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = Path(__file__).resolve().parent / "out"
AUDIO_EXTS = {".wav", ".mp3", ".m4a", ".ogg", ".flac", ".webm", ".mp4"}

BREAK_S = 0.4       # gap inside Michael's turn that becomes [break]
LONG_BREAK_S = 1.2  # ... or [long-break]

# Fish TTS (S2) tags Michael might plausibly produce on a call. Anything else is dropped,
# because unknown bracket text can be read aloud by TTS.
TTS_TAGS = {
    "break", "long-break", "chuckling", "laughing", "sighing", "clear throat", "emphasis",
    "happy", "excited", "calm", "relaxed", "confident", "surprised", "curious", "sarcastic",
    "empathetic", "frustrated", "worried", "uncertain", "confused", "satisfied", "grateful",
    "in a hurry tone", "soft tone", "whispering",
}
# ASR cue spellings -> TTS tag (ASR has no fixed enum; extend as real cues show up).
CUE_TO_TAG = {
    "laughter": "laughing", "laugh": "laughing", "laughs": "laughing", "laughing": "laughing",
    "chuckle": "chuckling", "chuckles": "chuckling", "giggle": "chuckling",
    "sigh": "sighing", "sighs": "sighing", "cough": "clear throat", "throat clearing": "clear throat",
    "happy": "happy", "excited": "excited", "calm": "calm", "relaxed": "relaxed", "neutral": None,
    "surprised": "surprised", "curious": "curious", "sarcastic": "sarcastic", "frustrated": "frustrated",
    "sad": None, "angry": "frustrated", "confused": "confused", "worried": "worried", "uncertain": "uncertain",
    "whisper": "whispering", "whispering": "whispering", "hurried": "in a hurry tone", "emphasis": "emphasis",
}
FILLERS = {"um", "uh", "like", "yeah", "so", "you know", "i mean", "kinda", "sort of", "right"}

SPEAKER_RE = re.compile(r"<\|speaker:(\d+)\|>(.*?)(?=<\|speaker:\d+\|>|$)", re.DOTALL)
CUE_RE = re.compile(r"\[([^\[\]]{1,40})\]")
WORD_RE = re.compile(r"[a-z0-9']+")


def words(s: str) -> list[str]:
    return WORD_RE.findall(CUE_RE.sub(" ", s.lower()))


def map_cues(text: str) -> tuple[str, Counter]:
    """Rewrite ASR cues to renderable TTS tags; drop the rest."""
    counts: Counter = Counter()

    def sub(m: re.Match) -> str:
        cue = m.group(1).strip().lower()
        tag = cue if cue in TTS_TAGS else CUE_TO_TAG.get(cue)
        if tag:
            counts[tag] += 1
            return f"[{tag}]"
        return ""

    return re.sub(r"\s{2,}", " ", CUE_RE.sub(sub, text)).strip(), counts


@dataclass
class Turn:
    speaker: str
    text: str                                   # annotated, TTS tags only
    segments: list[dict] = field(default_factory=list)

    @property
    def start(self) -> float | None:
        return self.segments[0]["start"] if self.segments else None

    @property
    def end(self) -> float | None:
        return self.segments[-1]["end"] if self.segments else None


def split_turns(asr: dict) -> list[Turn]:
    text = asr.get("text", "")
    if "<|speaker:" not in text:
        text = "<|speaker:0|>" + text
    turns = []
    for m in SPEAKER_RE.finditer(text):
        body = m.group(2).strip()
        if words(body):
            turns.append(Turn(speaker=m.group(1), text=body))
    # merge consecutive same-speaker chunks
    merged: list[Turn] = []
    for t in turns:
        if merged and merged[-1].speaker == t.speaker:
            merged[-1].text += " " + t.text
        else:
            merged.append(t)
    return merged


def align_segments(turns: list[Turn], segments: list[dict]) -> None:
    """Assign each timed segment to the turn its words fall in (segments carry no speaker)."""
    bounds, total = [], 0
    for t in turns:
        n = len(words(t.text))
        bounds.append((total, total + n))
        total += n
    cursor = 0
    for seg in segments:
        n = len(words(seg.get("text", "")))
        if n == 0:
            continue
        mid = cursor + n / 2
        for t, (lo, hi) in zip(turns, bounds):
            if lo <= mid < hi or t is turns[-1]:
                t.segments.append(seg)
                break
        cursor += n


def insert_breaks(turn: Turn) -> str:
    """Put [break]/[long-break] after the segment that preceded a real pause."""
    text = turn.text
    for a, b in zip(turn.segments, turn.segments[1:]):
        gap = b["start"] - a["end"]
        if gap < BREAK_S:
            continue
        tag = "[long-break]" if gap >= LONG_BREAK_S else "[break]"
        tail = words(a.get("text", ""))[-3:]
        if not tail:
            continue
        # find the last words of segment a in the annotated text, after any earlier insertions
        pattern = r"\b" + r"\W+(?:\[[^\]]*\]\W*)*".join(map(re.escape, tail)) + r"\b[^\w\[]*"
        m = re.search(pattern, text, flags=re.IGNORECASE)
        if m and not text[m.end():].lstrip().startswith("["):
            text = text[: m.end()].rstrip() + f" {tag} " + text[m.end():].lstrip()
    return re.sub(r"\s{2,}", " ", text).strip()


def pick_me(turns: list[Turn], me: str) -> str:
    if me != "auto":
        return me
    talk = Counter()
    for t in turns:
        talk[t.speaker] += len(words(t.text))
    return talk.most_common(1)[0][0]


def build_example(asr: dict, me: str, system_prompt: str) -> tuple[dict | None, dict]:
    turns = split_turns(asr)
    align_segments(turns, asr.get("segments") or [])
    me = pick_me(turns, me) if turns else me
    messages = [{"role": "system", "content": system_prompt}]
    stats = {"reply_gaps": [], "turn_words": [], "wpm": [], "fillers": 0, "words": 0, "tags": Counter()}
    prev: Turn | None = None
    for t in turns:
        mapped, tag_counts = map_cues(t.text)
        t.text = mapped
        if t.speaker == me:
            content = insert_breaks(t)
            stats["tags"].update(tag_counts)
            stats["tags"].update(re.findall(r"\[(break|long-break)\]", content))
            w = words(content)
            stats["turn_words"].append(len(w))
            stats["words"] += len(w)
            joined = " ".join(w)
            stats["fillers"] += sum(len(re.findall(rf"\b{re.escape(f)}\b", joined)) for f in FILLERS)
            if t.start is not None and t.end and t.end > t.start:
                stats["wpm"].append(len(w) / ((t.end - t.start) / 60))
            if prev is not None and prev.speaker != me and prev.end is not None and t.start is not None:
                stats["reply_gaps"].append(round(t.start - prev.end, 3))
            role = "assistant"
        else:
            content = CUE_RE.sub("", mapped).strip()  # the caller's side stays plain text
            role = "user"
        if messages[-1]["role"] == role:
            messages[-1]["content"] += " " + content
        else:
            messages.append({"role": role, "content": content})
        prev = t
    has_pair = any(m["role"] == "user" for m in messages) and any(m["role"] == "assistant" for m in messages)
    if messages[1:] and messages[1]["role"] == "assistant":
        messages.insert(1, {"role": "user", "content": "(call connects)"})
    return ({"messages": messages} if has_pair else None), stats


def transcribe(path: Path, key: str) -> dict:
    import httpx

    cache = OUT / "asr_cache" / (path.stem + ".json")
    if cache.exists():
        return json.loads(cache.read_text())
    with path.open("rb") as f:
        r = httpx.post(
            "https://api.fish.audio/v1/asr",
            headers={"Authorization": f"Bearer {key}", "model": "transcribe-1-pro"},
            files={"audio": (path.name, f)},
            data={"ignore_timestamps": "false", "language": "en"},
            timeout=300,
        )
    if r.is_error:
        sys.exit(f"Fish ASR {r.status_code} on {path.name}: {r.text[:300]}")
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(r.text)
    return r.json()


def summarize(all_stats: list[dict]) -> dict:
    gaps = [g for s in all_stats for g in s["reply_gaps"] if -1 < g < 5]
    turns = [n for s in all_stats for n in s["turn_words"]]
    wpm = [x for s in all_stats for x in s["wpm"] if 40 < x < 400]
    words_total = sum(s["words"] for s in all_stats) or 1
    tags = sum((s["tags"] for s in all_stats), Counter())
    med = lambda xs: round(statistics.median(xs), 2) if xs else None
    return {
        "reply_gap_s_median": med(gaps),
        "reply_gap_s_p25": round(statistics.quantiles(gaps, n=4)[0], 2) if len(gaps) >= 4 else None,
        "words_per_min_median": med(wpm),
        "turn_words_median": med(turns),
        "fillers_per_100_words": round(100 * sum(s["fillers"] for s in all_stats) / words_total, 1),
        "tag_counts": dict(tags.most_common()),
        "my_turns": len(turns),
    }


def main() -> None:
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
    sys.path.insert(0, str(ROOT))
    from call_context import DEFAULT_CONTEXT

    p = argparse.ArgumentParser()
    p.add_argument("recordings", type=Path)
    p.add_argument("--me-speaker", default="auto", help="Fish speaker label for Michael, or 'auto'")
    a = p.parse_args()

    files = sorted(f for f in a.recordings.rglob("*") if f.suffix.lower() in AUDIO_EXTS)
    if not files:
        sys.exit(f"no audio files in {a.recordings}")
    key = os.environ["FISH_API_KEY"]
    system_prompt = DEFAULT_CONTEXT.render_prompt()
    OUT.mkdir(exist_ok=True)
    examples, all_stats = 0, []
    with (OUT / "sft.jsonl").open("w") as out:
        for f in files:
            example, stats = build_example(transcribe(f, key), a.me_speaker, system_prompt)
            all_stats.append(stats)
            if example:
                out.write(json.dumps(example) + "\n")
                examples += 1
            print(f"{f.name}: {len(stats['turn_words'])} of your turns, tags {dict(stats['tags'])}")
    cadence = summarize(all_stats)
    (OUT / "cadence.json").write_text(json.dumps(cadence, indent=2))
    print(f"\n{examples} training conversations -> {OUT / 'sft.jsonl'}")
    print(json.dumps(cadence, indent=2))


if __name__ == "__main__":
    main()
