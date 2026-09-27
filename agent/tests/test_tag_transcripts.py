"""Offline tests for training/tag_transcripts.py using a Fish transcribe-1-pro-shaped response."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "training"))

from tag_transcripts import build_example, map_cues, summarize  # noqa: E402

ASR = {
    "text": (
        "<|speaker:0|>Hey Michael, where are you at?"
        "<|speaker:1|>[laughter] Yeah so I'm driving right now, I'll be there in like ten minutes."
        " Grabbing coffee with Sam at twelve thirty.[neutral]"
        "<|speaker:0|>Cool, see you soon."
        "<|speaker:1|>[sigh] Yep, see you."
    ),
    "segments": [
        {"text": "Hey Michael, where are you at?", "start": 0.0, "end": 1.6},
        {"text": "Yeah so I'm driving right now,", "start": 2.1, "end": 3.4},
        {"text": "I'll be there in like ten minutes.", "start": 3.9, "end": 5.4},
        {"text": "Grabbing coffee with Sam at twelve thirty.", "start": 6.9, "end": 8.8},
        {"text": "Cool, see you soon.", "start": 9.2, "end": 10.0},
        {"text": "Yep, see you.", "start": 10.3, "end": 10.9},
    ],
}


def test_map_cues_keeps_renderable_tags_only():
    text, counts = map_cues("[laughter] hi [neutral] there [made-up cue] ok [chuckling]")
    assert text == "[laughing] hi there ok [chuckling]"
    assert counts == {"laughing": 1, "chuckling": 1}


def test_michael_turns_become_tagged_assistant_turns():
    example, stats = build_example(ASR, "1", "SYSTEM")
    roles = [m["role"] for m in example["messages"]]
    assert roles == ["system", "user", "assistant", "user", "assistant"]

    first = example["messages"][2]["content"]
    assert first.startswith("[laughing] Yeah so I'm driving right now,")
    # 0.5 s gap -> [break]; 1.5 s gap -> [long-break]
    assert "right now, [break] I'll be there" in first
    assert "ten minutes. [long-break] Grabbing coffee" in first
    assert "neutral" not in first

    assert example["messages"][4]["content"] == "[sighing] Yep, see you."
    # the caller's side is plain text
    assert "[" not in example["messages"][1]["content"]

    assert stats["reply_gaps"] == [0.5, 0.3]
    assert stats["tags"]["break"] == 1 and stats["tags"]["long-break"] == 1


def test_auto_picks_the_speaker_who_talks_most():
    example, _ = build_example(ASR, "auto", "SYSTEM")
    assert example["messages"][2]["role"] == "assistant"
    assert example["messages"][2]["content"].startswith("[laughing] Yeah so")


def test_monologue_without_other_speaker_is_skipped():
    example, stats = build_example({"text": "just me talking", "segments": []}, "auto", "SYSTEM")
    assert example is None
    assert stats["turn_words"] == [3]


def test_cadence_summary():
    _, stats = build_example(ASR, "1", "SYSTEM")
    c = summarize([stats])
    assert c["reply_gap_s_median"] == 0.4
    assert c["my_turns"] == 2
    assert c["tag_counts"]["laughing"] == 1
