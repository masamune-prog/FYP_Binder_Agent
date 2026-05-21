#!/usr/bin/env python3
"""Entry point for the protein candidate design agent.

Usage::
    # Run the agent (with structural filters by default)
    uv run python main.py --target-fasta targets/egfr.fasta --target-name EGFR

    # With custom model and threshold
    uv run python main.py \\
        --target-fasta targets/egfr.fasta \\
        --target-name EGFR \\
        --model o3-mini \\
        --identity-threshold 0.80

    # Use different models for research and reasoning
    uv run python main.py \
        --target-fasta targets/egfr.fasta \
        --target-name EGFR \
        --research-model gpt-4.1-mini \
        --reasoning-model o3-mini

    # Skip structural filters
    uv run python main.py --target-fasta targets/egfr.fasta \\
        --target-name EGFR --no-af2ig --no-protenix

    # Inspect traces afterwards
    uv run python -m core.inspector --latest
"""

from __future__ import annotations

import json
import sys
from argparse import ArgumentParser
from pathlib import Path

from core.agent import AgentConfig, load_fasta_file, run_agent


def build_parser() -> ArgumentParser:
    parser = ArgumentParser(
        description="Run the smolagents protein candidate design agent",
        epilog="Requires OPENAI_API_KEY in environment.",
    )
    parser.add_argument(
        "--target-fasta",
        required=True,
        help="Path to the target protein FASTA file",
    )
    parser.add_argument(
        "--target-name",
        default=None,
        help="Name of the target protein (e.g. EGFR, HER2). Used for database lookups.",
    )
    parser.add_argument(
        "--identity-threshold",
        type=float,
        default=0.80,
        help="Maximum allowed sequence identity to known binders (default: 0.80)",
    )
    parser.add_argument(
        "--model",
        default="o3-mini",
        help="Default model ID to use for both agents unless overridden.",
    )
    parser.add_argument(
        "--research-model",
        default=None,
        help="Model ID to use for the research agent (defaults to --model).",
    )
    parser.add_argument(
        "--reasoning-model",
        default=None,
        help="Model ID to use for the reasoning agent (defaults to --model).",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional path to write JSON output (default: stdout only)",
    )

    # ── Structural filter flags ──────────────────────────────────────
    struct_group = parser.add_argument_group("structural validation filters")
    struct_group.add_argument(
        "--no-af2ig",
        action="store_true",
        help="Skip AF2-IG (ColabFold multimer) structural validation",
    )
    struct_group.add_argument(
        "--no-protenix",
        action="store_true",
        help="Skip Protenix structural validation",
    )
    struct_group.add_argument(
        "--af2ig-threshold",
        type=float,
        default=0.6,
        help="Minimum ipTM score to pass AF2-IG filter (default: 0.6)",
    )
    struct_group.add_argument(
        "--protenix-threshold",
        type=float,
        default=0.7,
        help="Minimum ipTM score to pass Protenix filter (default: 0.7)",
    )
    struct_group.add_argument(
        "--protenix-seeds",
        type=int,
        default=3,
        help="Number of Protenix prediction seeds (default: 3)",
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    # Load target FASTA
    target_path = Path(args.target_fasta)
    if not target_path.exists():
        print(f"ERROR: Target FASTA file not found: {target_path}", file=sys.stderr)
        return 1

    target_fasta = load_fasta_file(str(target_path))

    # Build config
    config = AgentConfig(
        target_fasta=target_fasta,
        target_name=args.target_name,
        identity_threshold=args.identity_threshold,
        model_id=args.model,
        research_model_id=args.research_model,
        reasoning_model_id=args.reasoning_model,
    )

    # Run agent
    try:
        result_json = run_agent(config)
    except EnvironmentError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    # Output agent results
    print("\n" + "=" * 60)
    print("AGENT OUTPUT")
    print("=" * 60)
    print(result_json)

    # ── Post-pipeline structural validation ──────────────────────────
    enable_af2ig = not args.no_af2ig
    enable_protenix = not args.no_protenix

    if enable_af2ig or enable_protenix:
        try:
            parsed = json.loads(result_json)
        except (json.JSONDecodeError, TypeError):
            parsed = None

        if parsed and parsed.get("decision") == "accepted":
            result_json = _run_structural_filters(
                parsed,
                enable_af2ig=enable_af2ig,
                enable_protenix=enable_protenix,
                af2ig_threshold=args.af2ig_threshold,
                protenix_threshold=args.protenix_threshold,
                protenix_seeds=args.protenix_seeds,
            )

            print("\n" + "=" * 60)
            print("FINAL OUTPUT (after structural validation)")
            print("=" * 60)
            print(result_json)
        elif parsed:
            print("\n[structural] Skipping structural validation — "
                  f"candidate was {parsed.get('decision', 'unknown')}")
    else:
        print("\n[structural] All structural filters disabled via CLI flags.")

    # Save to file if requested
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(result_json, encoding="utf-8")
        print(f"\nOutput also saved to {output_path}")

    return 0


def _run_structural_filters(
    parsed: dict,
    *,
    enable_af2ig: bool,
    enable_protenix: bool,
    af2ig_threshold: float,
    protenix_threshold: float,
    protenix_seeds: int,
) -> str:
    """Run structural filters on an accepted candidate and update the result."""
    from core.structural_filters import run_structural_validation

    candidate_fasta = parsed.get("candidate_seq_fasta", "")
    target_fasta = parsed.get("target_seq_fasta", "")

    if not candidate_fasta or not target_fasta:
        print("[structural] Missing FASTA sequences — skipping filters.")
        return json.dumps(parsed, indent=2, default=str)

    results = run_structural_validation(
        candidate_fasta=candidate_fasta,
        target_fasta=target_fasta,
        enable_af2ig=enable_af2ig,
        enable_protenix=enable_protenix,
        af2ig_threshold=af2ig_threshold,
        protenix_threshold=protenix_threshold,
        protenix_seeds=protenix_seeds,
    )

    # Add structural results to output
    parsed["structural_validation"] = [r.to_dict() for r in results]

    # If any enabled filter failed, reject the candidate
    if any(not r.passed for r in results):
        parsed["decision"] = "rejected_structural"
        failed = [r.filter_name for r in results if not r.passed]
        print(f"\n[structural] ❌ Candidate REJECTED by: {', '.join(failed)}")
    else:
        print(f"\n[structural] ✅ Candidate PASSED all structural filters!")

    return json.dumps(parsed, indent=2, default=str)


if __name__ == "__main__":
    raise SystemExit(main())
