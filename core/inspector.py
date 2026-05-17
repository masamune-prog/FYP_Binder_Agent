"""Inspect and analyse saved reasoning traces.

Usage::

    uv run python -m core.inspector                    # list all traces
    uv run python -m core.inspector --latest           # show the most recent trace
    uv run python -m core.inspector --file <path>      # show a specific trace
"""

from __future__ import annotations

import json
import sys
from argparse import ArgumentParser
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
TRACES_DIR = ROOT_DIR / "traces"


def load_all_traces(traces_dir: Path | None = None) -> list[dict]:
    """Load all trace JSON files, sorted newest-first."""
    d = traces_dir or TRACES_DIR
    if not d.exists():
        return []
    traces = []
    for f in sorted(d.glob("*.json"), reverse=True):
        try:
            traces.append(json.loads(f.read_text(encoding="utf-8")))
        except (json.JSONDecodeError, OSError):
            continue
    return traces


def summarize_trace(trace: dict) -> str:
    """Return a human-readable summary of a single trace."""
    lines = []
    kind = trace.get("kind", trace.get("type", "trace"))
    created = trace.get("created_at") or trace.get("saved_at", "?")
    lines.append(f"── {kind} @ {created} ──")

    if "model_id" in trace:
        lines.append(f"  Model:      {trace['model_id']}")
    if "target_name" in trace:
        lines.append(f"  Target:     {trace['target_name']}")
    if "identity_threshold" in trace:
        lines.append(f"  Threshold:  {trace['identity_threshold']}")

    # Agent execution log
    if "agent_logs" in trace:
        logs = trace["agent_logs"]
        if isinstance(logs, list):
            lines.append(f"  Log steps:  {len(logs)}")
        else:
            lines.append(f"  Log type:   {type(logs).__name__}")

    # Reasoning trace entries
    if "entries" in trace:
        for entry in trace["entries"]:
            step = entry.get("step", "?")
            summary = entry.get("summary", "")
            lines.append(f"  [{step}] {summary}")
            meta = entry.get("metadata", {})
            if meta:
                for k, v in meta.items():
                    lines.append(f"    {k}: {v}")

    # Inline trace fields (from agent-saved traces)
    for key in ("plan", "decision", "reasoning"):
        if key in trace:
            val = str(trace[key])[:200]
            lines.append(f"  {key}: {val}")

    if "verification" in trace:
        v = trace["verification"]
        if isinstance(v, dict):
            for k, val in v.items():
                lines.append(f"  verification.{k}: {val}")

    return "\n".join(lines)


def list_trace_files(traces_dir: Path | None = None) -> list[Path]:
    """Return trace file paths sorted newest-first."""
    d = traces_dir or TRACES_DIR
    if not d.exists():
        return []
    return sorted(d.glob("*.json"), reverse=True)


def main(argv: list[str] | None = None) -> int:
    parser = ArgumentParser(description="Inspect protein candidate reasoning traces")
    parser.add_argument("--latest", action="store_true", help="Show only the most recent trace")
    parser.add_argument("--file", type=str, default=None, help="Path to a specific trace file")
    parser.add_argument("--json", action="store_true", help="Output raw JSON instead of summary")
    args = parser.parse_args(argv)

    if args.file:
        path = Path(args.file)
        if not path.exists():
            print(f"File not found: {path}", file=sys.stderr)
            return 1
        trace = json.loads(path.read_text(encoding="utf-8"))
        if args.json:
            print(json.dumps(trace, indent=2))
        else:
            print(summarize_trace(trace))
        return 0

    files = list_trace_files()
    if not files:
        print("No traces found in", TRACES_DIR)
        return 0

    if args.latest:
        trace = json.loads(files[0].read_text(encoding="utf-8"))
        print(f"File: {files[0]}\n")
        if args.json:
            print(json.dumps(trace, indent=2))
        else:
            print(summarize_trace(trace))
        return 0

    print(f"Found {len(files)} trace(s) in {TRACES_DIR}\n")
    for f in files:
        try:
            trace = json.loads(f.read_text(encoding="utf-8"))
            print(f"File: {f.name}")
            print(summarize_trace(trace))
            print()
        except (json.JSONDecodeError, OSError) as exc:
            print(f"File: {f.name}  [ERROR: {exc}]")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
