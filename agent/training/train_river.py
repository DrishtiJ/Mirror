"""LoRA SFT on River: teach the base model the speaker's phrasing and cadence.

Input: JSONL, one conversation per line: {"messages": [...], "tools"?: [...]} (OpenAI-style messages;
tool rows come from make_tool_examples.py). Pass several files to mix style data with tool examples.
The assistant turns are the speaker we're cloning; every assistant turn gets loss (ALL_ASSISTANT).
The system message should be the production prompt (call_context.render_prompt) so training
matches what the live agent sends.

    uv run training/train_river.py training/out/sft.jsonl --dry-run      # render + token stats only
    uv run training/train_river.py training/out/sft.jsonl                # train, save, print checkpoint

Writes training/out/train_run.json with the checkpoint paths (inference checkpoint is what a
dedicated deployment serves).
"""

import argparse
import json
import os
import random
import statistics
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

import river_client as river  # noqa: E402
from river_client.renderers import TrainOnWhat, get_renderer  # noqa: E402

BASE_MODEL = os.getenv("RIVER_BASE_MODEL", "deepseek-ai/DeepSeek-V4.1-Flash")
OUT = Path(__file__).resolve().parent / "out"


def load(paths: list[Path]) -> list[dict]:
    """Rows of {"messages": [...], "tools"?: [...]} from one or more JSONL files."""
    rows = []
    for path in paths:
        for line in path.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                if any(m["role"] == "assistant" and (m.get("content") or m.get("tool_calls")) for m in row["messages"]):
                    rows.append(row)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("data", type=Path, nargs="+", help="JSONL files, e.g. style data + tool_sft.jsonl")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--max-length", type=int, default=8192)
    ap.add_argument("--name", default=f"cadence-{time.strftime('%Y%m%d-%H%M')}")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    rows = load(args.data)
    renderer = get_renderer(BASE_MODEL, thinking=False)
    data = [renderer.build_training_example(r["messages"], train_on=TrainOnWhat.ALL_ASSISTANT,
                                            max_length=args.max_length, tools=r.get("tools")).to_dict()
            for r in rows]
    lengths = [len(d["input_ids"]) for d in data]
    print(f"{len(data)} conversations, tokens: total {sum(lengths)}, median {statistics.median(lengths):.0f}, "
          f"max {max(lengths)}; base {BASE_MODEL}; {args.epochs} epochs x batch {args.batch}, lr {args.lr}, rank {args.rank}")
    if args.dry_run:
        return

    client = river.Client(api_key=os.environ["RIVER_API_KEY"])
    print("connecting to River, health:", client.health_check(), flush=True)
    run: dict = {"name": args.name, "base_model": BASE_MODEL, "examples": len(data), "args": vars(args) | {"data": [str(p) for p in args.data]},
                 "losses": []}
    print("requesting a training session (GPU allocation)...", flush=True)
    with client.session(experiment="voice-cadence", run=args.name) as session:
        print("session:", session.session_id, "- creating LoRA model...", flush=True)
        model = session.create_model(base_model=BASE_MODEL, lora=river.LoraConfig(rank=args.rank))
        run["training_run_id"] = model.training_run_id
        print("training run:", model.training_run_id, flush=True)
        rng = random.Random(0)
        for epoch in range(args.epochs):
            order = list(range(len(data)))
            rng.shuffle(order)
            for i in range(0, len(order), args.batch):
                batch = [data[j] for j in order[i:i + args.batch]]
                fb, opt = model.train_step(batch, lr=args.lr, loss_fn="cross_entropy", grad_clip_norm=1.0)
                loss = fb.metrics.get("loss_mean")
                run["losses"].append({"step": model.step, "epoch": epoch, "loss": loss})
                print(f"epoch {epoch} step {model.step} loss {loss:.4f} grad_norm {opt.metrics.get('grad_norm')}", flush=True)
            model.save_weights(f"{args.name}-epoch{epoch}")  # training-mode checkpoint (resumable)
        ckpt = model.save_weights(f"{args.name}-serving", mode="inference")
        run["inference_checkpoint"] = ckpt.path
        print("inference checkpoint:", ckpt.path)

    OUT.mkdir(exist_ok=True)
    (OUT / f"train_run_{args.name}.json").write_text(json.dumps(run, indent=2))
    print(f"saved {OUT / f'train_run_{args.name}.json'}")


if __name__ == "__main__":
    sys.exit(main())
