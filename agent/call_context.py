"""Per-call context rendered into the system prompt.

The rendered prompt must match what River was trained on, so the only thing
that varies per call is the placeholder values.
"""

import json
import os
import logging
from dataclasses import asdict, dataclass
from pathlib import Path

logger = logging.getLogger("call_context")

PROMPT_TEMPLATE = (Path(__file__).parent / "system_prompt.txt").read_text()


@dataclass(frozen=True)
class CallContext:
    speaker_name: str
    caller_name: str
    notes: str

    def render_prompt(self) -> str:
        return PROMPT_TEMPLATE.format(**asdict(self))


DEFAULT_CONTEXT = CallContext(
    speaker_name=os.getenv("SPEAKER_NAME", "Michael"),
    caller_name="the caller",
    notes="No prior context for this call.",
)


def from_metadata(*sources: str | None) -> CallContext:
    """First source that parses as a JSON object wins; missing keys fall back to DEFAULT_CONTEXT.

    Sources are job metadata, then room metadata, as JSON:
    {"speaker_name": "...", "caller_name": "...", "notes": "..."}.
    """
    for raw in sources:
        if not raw:
            continue
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            logger.warning("ignoring non-JSON call metadata: %r", raw[:200])
            continue
        if not isinstance(data, dict):
            continue
        fields = {k: str(data[k]).strip() for k in asdict(DEFAULT_CONTEXT) if data.get(k)}
        return CallContext(**{**asdict(DEFAULT_CONTEXT), **fields})
    return DEFAULT_CONTEXT
