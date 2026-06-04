"""Post-pipeline structural validation filters.

Runs AF2-IG (via ColabFold) and/or Protenix to predict whether a designed
candidate binder will actually fold and bind to the target protein.  Both
tools are invoked via subprocess to avoid dependency conflicts between
JAX (ColabFold) and PyTorch (Protenix).
"""

from __future__ import annotations

import glob
import json
import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parent.parent
STRUCTURAL_DIR = ROOT_DIR / "traces" / "structural"


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class FilterResult:
    """Outcome of a single structural validation filter."""

    filter_name: str  # "AF2-IG" or "Protenix"
    passed: bool
    iptm: float
    ptm: float | None = None
    plddt: float | None = None
    threshold: float = 0.0
    output_dir: str = ""
    raw_scores: dict = field(default_factory=dict)
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_FASTA_HEADER = re.compile(r"^>.*", re.MULTILINE)


def _extract_sequence(fasta: str) -> str:
    """Extract the raw amino-acid sequence from a FASTA string."""
    lines = fasta.strip().splitlines()
    return "".join(
        line.strip() for line in lines if not line.startswith(">")
    ).upper()


def _ensure_tool(name: str) -> str:
    """Return the path to a CLI tool or raise with a helpful message."""
    path = shutil.which(name)
    if path is None:
        raise FileNotFoundError(
            f"'{name}' not found on PATH. "
            f"Install it with: pip install -e '.[structural]'"
        )
    return path


def _make_output_dir(filter_name: str) -> Path:
    """Create a timestamped output directory under traces/structural/."""
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out = STRUCTURAL_DIR / ts / filter_name
    out.mkdir(parents=True, exist_ok=True)
    return out


# ---------------------------------------------------------------------------
# AF2-IG filter  (ColabFold multimer)
# ---------------------------------------------------------------------------

def run_af2ig_filter(
    candidate_fasta: str,
    target_fasta: str,
    iptm_threshold: float = 0.6,
) -> FilterResult:
    """Run AlphaFold2 Initial-Guess filter via ColabFold multimer.

    Creates a combined FASTA with both chains separated by ``:``, runs the
    ColabFold Python API (``get_queries`` + ``run``), and parses the
    resulting scores for ipTM.

    Args:
        candidate_fasta: Candidate binder in FASTA format.
        target_fasta: Target protein in FASTA format.
        iptm_threshold: Minimum ipTM to pass (default 0.6).

    Returns:
        FilterResult with pass/fail and confidence scores.
    """
    output_dir = _make_output_dir("af2ig")

    candidate_seq = _extract_sequence(candidate_fasta)
    target_seq = _extract_sequence(target_fasta)

    # ColabFold multimer expects chains separated by ':' in a single FASTA entry.
    combined_seq = f"{candidate_seq}:{target_seq}"
    input_dir = output_dir / "tmp_input"
    input_dir.mkdir(parents=True, exist_ok=True)
    input_fasta = input_dir / "candidate_target_complex.fasta"
    input_fasta.write_text(
        f">candidate_target_complex\n{combined_seq}\n",
        encoding="utf-8",
    )

    print(f"[AF2-IG] Running ColabFold multimer prediction...")
    print(f"[AF2-IG] Candidate length: {len(candidate_seq)}")
    print(f"[AF2-IG] Target length: {len(target_seq)}")
    print(f"[AF2-IG] Output: {output_dir}")

    try:
        # Import inside the function so the module can still be imported without ColabFold.
        from colabfold.batch import get_queries, run
        from colabfold.utils import setup_logging
    except Exception as exc:
        return FilterResult(
            filter_name="AF2-IG",
            passed=False,
            iptm=0.0,
            threshold=iptm_threshold,
            output_dir=str(output_dir),
            error=(
                "ColabFold Python API import failed. Install ColabFold in this "
                f"environment. Details: {exc}"
            ),
        )

    try:
        setup_logging(output_dir / "log.txt")
        queries, is_complex = get_queries(str(input_dir))

        # Match the tested invocation defaults from the working script.
        run(
            queries=queries,
            result_dir=str(output_dir),
            is_complex=is_complex,
            use_bfloat16=False,
            use_templates=False,
            msa_mode="MMseqs2 (UniRef+Environmental)",
            model_type="alphafold2_multimer_v3",
            num_models=1,
            num_recycles=3,
            num_relax=1,
            relax_max_iterations=2000,
        )
    except subprocess.TimeoutExpired:
        return FilterResult(
            filter_name="AF2-IG",
            passed=False,
            iptm=0.0,
            threshold=iptm_threshold,
            output_dir=str(output_dir),
            error="ColabFold timed out",
        )
    except Exception as exc:
        return FilterResult(
            filter_name="AF2-IG",
            passed=False,
            iptm=0.0,
            threshold=iptm_threshold,
            output_dir=str(output_dir),
            error=f"ColabFold run failed: {exc}",
        )
    finally:
        try:
            input_fasta.unlink(missing_ok=True)
        except OSError:
            pass

    # Parse scores — ColabFold writes *_scores_rank_*.json
    scores = _parse_colabfold_scores(output_dir)

    iptm = scores.get("iptm", 0.0)
    plddt = scores.get("mean_plddt", None)
    passed = iptm >= iptm_threshold

    print(f"[AF2-IG] ipTM: {iptm:.3f} (threshold: {iptm_threshold})")
    print(f"[AF2-IG] pLDDT: {plddt}")
    print(f"[AF2-IG] Result: {'PASS ✅' if passed else 'FAIL ❌'}")

    return FilterResult(
        filter_name="AF2-IG",
        passed=passed,
        iptm=iptm,
        plddt=plddt,
        threshold=iptm_threshold,
        output_dir=str(output_dir),
        raw_scores=scores,
    )


def _parse_colabfold_scores(output_dir: Path) -> dict:
    """Parse ColabFold score files and return the best-ranking scores."""
    # ColabFold outputs *_scores_rank_001_*.json
    score_files = sorted(glob.glob(str(output_dir / "*_scores_rank_*.json")))
    if not score_files:
        # Try alternative naming pattern
        score_files = sorted(glob.glob(str(output_dir / "*scores*.json")))

    if not score_files:
        return {"error": "No score files found", "iptm": 0.0}

    # Use the top-ranked file
    with open(score_files[0], encoding="utf-8") as f:
        data = json.load(f)

    scores: dict[str, Any] = {}

    # ColabFold score format varies; extract what we can
    if "iptm" in data:
        scores["iptm"] = float(data["iptm"])
    elif "iptm+ptm" in data:
        scores["iptm"] = float(data["iptm+ptm"])
    else:
        scores["iptm"] = 0.0

    if "ptm" in data:
        scores["ptm"] = float(data["ptm"])

    if "plddt" in data:
        plddt_vals = data["plddt"]
        if isinstance(plddt_vals, list):
            scores["mean_plddt"] = sum(plddt_vals) / len(plddt_vals)
        else:
            scores["mean_plddt"] = float(plddt_vals)
    elif "mean_plddt" in data:
        scores["mean_plddt"] = float(data["mean_plddt"])

    scores["source_file"] = score_files[0]
    return scores


# ---------------------------------------------------------------------------
# Protenix filter
# ---------------------------------------------------------------------------

def run_protenix_filter(
    candidate_fasta: str,
    target_fasta: str,
    iptm_threshold: float = 0.7,
    num_seeds: int = 3,
) -> FilterResult:
    """Run Protenix complex structure prediction.

    Creates a Protenix input JSON with both protein chains, runs
    ``protenix pred``, and parses the confidence JSON for ipTM.

    Args:
        candidate_fasta: Candidate binder in FASTA format.
        target_fasta: Target protein in FASTA format.
        iptm_threshold: Minimum ipTM to pass (default 0.7).
        num_seeds: Number of prediction seeds (default 3).

    Returns:
        FilterResult with pass/fail and confidence scores.
    """
    protenix_bin = _ensure_tool("protenix")
    output_dir = _make_output_dir("protenix")

    candidate_seq = _extract_sequence(candidate_fasta)
    target_seq = _extract_sequence(target_fasta)

    # Build Protenix input JSON
    input_data = [
        {
            "sequences": [
                {"protein": {"id": "A", "sequence": candidate_seq}},
                {"protein": {"id": "B", "sequence": target_seq}},
            ],
            "name": "candidate_target_complex",
        }
    ]

    input_json = output_dir / "input.json"
    input_json.write_text(
        json.dumps(input_data, indent=2), encoding="utf-8"
    )

    seeds = ",".join(str(101 + i) for i in range(num_seeds))

    print(f"[Protenix] Running complex structure prediction...")
    print(f"[Protenix] Candidate length: {len(candidate_seq)}")
    print(f"[Protenix] Target length: {len(target_seq)}")
    print(f"[Protenix] Seeds: {seeds}")
    print(f"[Protenix] Output: {output_dir}")

    try:
        result = subprocess.run(
            [
                protenix_bin, "predict",
                "--inputs", str(input_json),
                "--output", str(output_dir),
                "--seeds", seeds,
            ],
            capture_output=True,
            text=True,
            timeout=3600,  # 60 min timeout
        )

        if result.returncode != 0:
            return FilterResult(
                filter_name="Protenix",
                passed=False,
                iptm=0.0,
                threshold=iptm_threshold,
                output_dir=str(output_dir),
                error=f"protenix pred failed (rc={result.returncode}): "
                      f"{result.stderr[-500:] if result.stderr else 'no stderr'}",
            )

    except subprocess.TimeoutExpired:
        return FilterResult(
            filter_name="Protenix",
            passed=False,
            iptm=0.0,
            threshold=iptm_threshold,
            output_dir=str(output_dir),
            error="Protenix timed out after 60 minutes",
        )

    # Parse confidence scores
    scores = _parse_protenix_scores(output_dir)

    iptm = scores.get("iptm", 0.0)
    ptm = scores.get("ptm", None)
    plddt = scores.get("plddt", None)
    passed = iptm >= iptm_threshold

    print(f"[Protenix] ipTM: {iptm:.3f} (threshold: {iptm_threshold})")
    print(f"[Protenix] pTM: {ptm}")
    print(f"[Protenix] pLDDT: {plddt}")
    print(f"[Protenix] Result: {'PASS ✅' if passed else 'FAIL ❌'}")

    return FilterResult(
        filter_name="Protenix",
        passed=passed,
        iptm=iptm,
        ptm=ptm,
        plddt=plddt,
        threshold=iptm_threshold,
        output_dir=str(output_dir),
        raw_scores=scores,
    )


def _parse_protenix_scores(output_dir: Path) -> dict:
    """Parse Protenix confidence JSON files and return the best scores.

    Protenix uses ranking_score = 0.8*ipTM + 0.2*pTM - 100*has_clash
    to rank predictions.  We pick the sample with the highest ranking_score.
    """
    conf_files = sorted(
        glob.glob(str(output_dir / "**/*summary_confidence*.json"), recursive=True)
    )

    if not conf_files:
        return {"error": "No confidence files found", "iptm": 0.0}

    best_score = -float("inf")
    best_data: dict[str, Any] = {}

    for fpath in conf_files:
        try:
            with open(fpath, encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue

        iptm = float(data.get("iptm", 0.0))
        ptm = float(data.get("ptm", 0.0))
        has_clash = float(data.get("has_clash", 0.0))
        ranking = 0.8 * iptm + 0.2 * ptm - 100.0 * has_clash

        if ranking > best_score:
            best_score = ranking
            best_data = data
            best_data["_source_file"] = fpath
            best_data["_ranking_score"] = ranking

    if not best_data:
        return {"error": "Could not parse any confidence files", "iptm": 0.0}

    scores: dict[str, Any] = {
        "iptm": float(best_data.get("iptm", 0.0)),
        "ptm": float(best_data.get("ptm", 0.0)),
        "ranking_score": best_data.get("_ranking_score", 0.0),
        "source_file": best_data.get("_source_file", ""),
    }

    # pLDDT may be per-residue or a scalar
    plddt = best_data.get("plddt")
    if isinstance(plddt, list):
        scores["plddt"] = sum(plddt) / len(plddt) if plddt else None
    elif plddt is not None:
        scores["plddt"] = float(plddt)

    # Chain-pair ipTM matrix if available
    if "chain_pair_iptm" in best_data:
        scores["chain_pair_iptm"] = best_data["chain_pair_iptm"]

    return scores


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def run_structural_validation(
    candidate_fasta: str,
    target_fasta: str,
    *,
    enable_af2ig: bool = True,
    enable_protenix: bool = True,
    af2ig_threshold: float = 0.6,
    protenix_threshold: float = 0.7,
    protenix_seeds: int = 3,
) -> list[FilterResult]:
    """Run all enabled structural validation filters.

    Returns a list of FilterResult objects. The candidate is considered
    rejected if ANY enabled filter fails (strict mode).

    Args:
        candidate_fasta: Candidate binder in FASTA format.
        target_fasta: Target protein in FASTA format.
        enable_af2ig: Whether to run AF2-IG filter.
        enable_protenix: Whether to run Protenix filter.
        af2ig_threshold: ipTM threshold for AF2-IG.
        protenix_threshold: ipTM threshold for Protenix.
        protenix_seeds: Number of Protenix prediction seeds.
    """
    results: list[FilterResult] = []

    if enable_af2ig:
        print("\n" + "=" * 60)
        print("STRUCTURAL FILTER: AF2-IG (ColabFold)")
        print("=" * 60)
        try:
            af2ig_result = run_af2ig_filter(
                candidate_fasta, target_fasta, af2ig_threshold
            )
            results.append(af2ig_result)
        except FileNotFoundError as exc:
            print(f"[AF2-IG] Skipped: {exc}")
            results.append(FilterResult(
                filter_name="AF2-IG",
                passed=False,
                iptm=0.0,
                threshold=af2ig_threshold,
                error=str(exc),
            ))

    if enable_protenix:
        print("\n" + "=" * 60)
        print("STRUCTURAL FILTER: Protenix")
        print("=" * 60)
        try:
            protenix_result = run_protenix_filter(
                candidate_fasta, target_fasta, protenix_threshold, protenix_seeds
            )
            results.append(protenix_result)
        except FileNotFoundError as exc:
            print(f"[Protenix] Skipped: {exc}")
            results.append(FilterResult(
                filter_name="Protenix",
                passed=False,
                iptm=0.0,
                threshold=protenix_threshold,
                error=str(exc),
            ))

    return results
