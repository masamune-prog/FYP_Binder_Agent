#!/usr/bin/env python3
"""Run all structural filters on the scFv binder candidate.

This script imports the structural filters from core.structural_filters
and runs them against the provided binder/target pair. It exercises every
available filter: Chemical (RDKit), ESM2 (Hugging Face), AF2-IG (ColabFold),
AF2-PAE, Protenix, and Rosetta ΔSASA.

Filters whose dependencies are not installed will report an error gracefully
without crashing the pipeline.

Usage:
    python tests/test_structural_filters_hf.py
"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path

# Ensure project root is importable
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.structural_filters import (
    FilterResult,
    run_chemical_filter,
    run_esm2_ll_filter,
    run_rosetta_sasa_filter,
    run_af2_pae_filter,
    run_af2ig_filter,
    run_protenix_filter,
    run_structural_validation,
)


# ---------------------------------------------------------------------------
# Sequences
# ---------------------------------------------------------------------------

# scFv binder candidate
CANDIDATE_SEQ = (
    "EVQLVQSGAEVKKRGSSVKVSCKSSGGTFSNYAINWVRQAPGQGLEWMGGIIPILGIANYAQKFQG"
    "RVTITTDESTSTAYMELSSLRSEDTAVYYCARGWGREQLAPHPSQYYYYYYGMDVWGQGTTVTVSS"
    "GGGGSGGGGSGGGGSEIVMTQSPGTPSLSPGERATLSCRASQSIRSTYLAWYQQKPGQAPRLLIY"
    "GASSRATGIPDRFSGSGSGTDFTLTISRLEPEDFAVYYCQQYGRSPSFGQGTKVEIK"
)
CANDIDATE_FASTA = f">scfv_binder\n{CANDIDATE_SEQ}\n"

# Target protein
TARGET_SEQ = (
    "MPAENKKVRFENTTSDKGKIPSKVIKSYYGTMDIKKINEGLLDSKILSAFNTVIALLGSIVIIVMNIMIIQ"
    "NYTRSTDNQAVIKDALQGIQQQIKGLADKIGTEIGPKVSLIDTSSTITIPANIGLLGSKISQSTASINENV"
    "NEKCKFTLPPLKIHECNISCPNPLPFREYRPQTEGVSNLVGLPNNICLQKTSNQILKPKLISYTLPVVGQ"
    "SGTCITDPLLAMDEGYFAYSHLERIGSCSRGVSKQRIIGVGEVLDRGDEVPSLFMTNVWTPPNPNTVYHCS"
    "AVYNNEFYYVLCAVSTVGDPILNSTYWSGSLMMTRLAVKPKSNGGGYNQHQLALRSIEKGRYDKVMPYGPS"
    "GIKQGDTLYFPAVGFLVRTEFKYNDSNCPITKCQYSKPENCRLSMGIRPNSHYILRSGLLKYNLSDGENP"
    "KVVFIEISDQRLSIGSPSKIYDSLGQPVFYQASFSWDTMIKFGDVLTVNPLVVNWRNNTVISRPGQSQCP"
    "RFNTCPEICWEGVYNDAFLIDRINWISAGVFLDSNQTAENPVFTVFKDNEILYRAQLASEDTNAQKTITNCF"
    "LLKNKIWCISLVEIYDTGDNVIRPKLFAVKIPEQCT"
)
TARGET_FASTA = f">target_protein\n{TARGET_SEQ}\n"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _banner(title: str) -> None:
    print(f"\n{'=' * 72}")
    print(f"  {title}")
    print(f"{'=' * 72}")


def _print_result(result: FilterResult) -> None:
    status = "✅ PASS" if result.passed else "❌ FAIL"
    print(f"  {result.filter_name:<28s}  {status}  score={result.iptm:.4f}  "
          f"threshold={result.threshold}")
    if result.error:
        print(f"    ⚠ Error: {result.error}")
    if result.output_dir:
        print(f"    📁 Output: {result.output_dir}")
    if result.raw_scores:
        # Print key metrics from raw_scores (skip large arrays)
        for k, v in result.raw_scores.items():
            if isinstance(v, list) and len(v) > 10:
                print(f"    {k}: [{len(v)} values]")
            else:
                print(f"    {k}: {v}")


def _print_summary(results: list[FilterResult]) -> None:
    print(f"\n{'=' * 72}")
    print(f"  {'Filter':<28s} {'Status':>8s} {'Score':>10s} {'Threshold':>10s}")
    print(f"  {'-' * 60}")
    for r in results:
        status = "✅ PASS" if r.passed else "❌ FAIL"
        score_str = f"{r.iptm:.4f}" if r.iptm != 0.0 else "N/A"
        thresh_str = f"{r.threshold:.4f}" if r.threshold != 0.0 else "N/A"
        print(f"  {r.filter_name:<28s} {status:>8s} {score_str:>10s} {thresh_str:>10s}")
        if r.error:
            print(f"    ⚠ {r.error[:100]}")
    print(f"{'=' * 72}")

    passed_count = sum(1 for r in results if r.passed)
    failed_count = sum(1 for r in results if not r.passed)
    error_count = sum(1 for r in results if r.error)
    print(f"\n  Total: {len(results)} filters  |  "
          f"✅ Passed: {passed_count}  |  "
          f"❌ Failed: {failed_count}  |  "
          f"⚠ Errors: {error_count}")


# ---------------------------------------------------------------------------
# Run individual filters
# ---------------------------------------------------------------------------

def main():
    print("=" * 72)
    print("🧬  STRUCTURAL FILTERS — FULL PIPELINE RUN")
    print("=" * 72)
    print(f"  Binder:  {CANDIDATE_SEQ[:50]}…  ({len(CANDIDATE_SEQ)} aa)")
    print(f"  Target:  {TARGET_SEQ[:50]}…  ({len(TARGET_SEQ)} aa)")

    all_results: list[FilterResult] = []

    # ── 1. Chemical filter (RDKit) ────────────────────────────────────────
    _banner("Filter 1: Chemical (RDKit Lipinski Ro5 + PAINS)")
    result = run_chemical_filter(CANDIDATE_FASTA)
    _print_result(result)
    all_results.append(result)

    # ── 2. ESM2 Log-Likelihood (Hugging Face) ─────────────────────────────
    _banner("Filter 2: ESM2 Log-Likelihood (Hugging Face)")
    result = run_esm2_ll_filter(CANDIDATE_FASTA)
    _print_result(result)
    all_results.append(result)

    # ── 3. Rosetta ΔSASA Docking ──────────────────────────────────────────
    _banner("Filter 3: Rosetta ΔSASA Docking")
    result = run_rosetta_sasa_filter(CANDIDATE_FASTA, TARGET_FASTA)
    _print_result(result)
    all_results.append(result)

    # ── 4. AF2-IG (ColabFold Multimer) ────────────────────────────────────
    _banner("Filter 4: AF2-IG (ColabFold Multimer iPTM)")
    try:
        result = run_af2ig_filter(CANDIDATE_FASTA, TARGET_FASTA)
    except FileNotFoundError as exc:
        result = FilterResult(
            filter_name="AF2-IG", passed=False, iptm=0.0,
            threshold=0.6, error=str(exc),
        )
    _print_result(result)
    all_results.append(result)
    af2ig_output = result.output_dir or None

    # ── 5. AF2-PAE (Inter-chain PAE + iPTM) ───────────────────────────────
    _banner("Filter 5: AF2-PAE (Inter-chain PAE + iPTM)")
    try:
        result = run_af2_pae_filter(
            CANDIDATE_FASTA,
            TARGET_FASTA,
            af2ig_output_dir=af2ig_output,
        )
    except (FileNotFoundError, Exception) as exc:
        result = FilterResult(
            filter_name="AF2-PAE", passed=False, iptm=0.0,
            threshold=0.6, error=str(exc),
        )
    _print_result(result)
    all_results.append(result)

    # ── 6. Protenix ───────────────────────────────────────────────────────
    _banner("Filter 6: Protenix (iPTM)")
    try:
        result = run_protenix_filter(CANDIDATE_FASTA, TARGET_FASTA)
    except FileNotFoundError as exc:
        result = FilterResult(
            filter_name="Protenix", passed=False, iptm=0.0,
            threshold=0.7, error=str(exc),
        )
    _print_result(result)
    all_results.append(result)

    # ── Summary ───────────────────────────────────────────────────────────
    _banner("RESULTS SUMMARY")
    _print_summary(all_results)

    # Save results to JSON
    output_path = ROOT / "traces" / "structural_filter_results.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps([asdict(r) for r in all_results], indent=2, default=str),
        encoding="utf-8",
    )
    print(f"\n  💾 Results saved to: {output_path}")

    # Return non-zero if any filter failed (ignoring dependency errors)
    real_failures = [
        r for r in all_results
        if not r.passed and not r.error
    ]
    return 1 if real_failures else 0


if __name__ == "__main__":
    sys.exit(main())
