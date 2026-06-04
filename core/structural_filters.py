"""Post-pipeline structural validation filters.

Runs AF2-IG (via ColabFold), Protenix, RDKit Chemical filters, ESM2 language model
log-likelihood, Rosetta dSASA docking, and AF2-PAE inter-chain metrics to predict
whether a designed candidate binder will actually fold and bind to the target protein.
"""

from __future__ import annotations

import glob
import json
import math
import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# RDKit imports (done dynamically inside functions to handle ImportError gracefully,
# but we define these names globally for type hinting/reference)
try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors
    from rdkit.Chem import Lipinski as RDKitLipinski
except ImportError:
    Chem = None
    Descriptors = None
    RDKitLipinski = None

try:
    from rdkit.Contrib.SA_Score import sascorer
except ImportError:
    sascorer = None

ROOT_DIR = Path(__file__).resolve().parent.parent
STRUCTURAL_DIR = ROOT_DIR / "traces" / "structural"


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class FilterResult:
    """Outcome of a single structural validation filter."""

    filter_name: str  # "AF2-IG", "Protenix", "Chemical (RDKit)", "ESM2 LL", etc.
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


def _is_peptide_sequence(seq: str) -> bool:
    """Check if a sequence consists of standard amino acids and is peptide-like (>5 aa)."""
    seq = seq.strip().upper()
    if not seq:
        return False
    # Standard amino acids + 'X' (any) + '*' (stop) + ':' (chain separator)
    aa_chars = set("ACDEFGHIKLMNPQRSTVWYX*:")
    return all(c in aa_chars for c in seq) and len(seq) > 5


def fasta_to_smiles(fasta_text: str) -> str:
    """Convert a single protein FASTA record into a peptide SMILES string."""
    if not isinstance(fasta_text, str) or not fasta_text.strip():
        raise ValueError("FASTA input must be a non-empty string.")

    from rdkit import Chem
    mol = Chem.MolFromFASTA(fasta_text)
    if mol is None:
        raise ValueError("RDKit could not convert the FASTA sequence to a molecule.")

    return Chem.MolToSmiles(mol)


def check_lipinski(mol: Chem.Mol, is_peptide: bool = False) -> bool:
    """Checks if a molecule satisfies Lipinski's Rule of 5.

    - Molecular Weight < 500 Da
    - LogP < 5
    - H-bond Donors < 5
    - H-bond Acceptors < 10

    Allows for one violation (standard relaxed criteria), but can be
    strictified by changing the allowed violations to 0.

    Checks Lipinski with relaxed thresholds if the molecule is known
    to be a peptide or macrocycle.
    """
    if mol is None:
        return False

    from rdkit.Chem import Descriptors
    from rdkit.Chem import Lipinski as RDKitLipinski

    mw = Descriptors.MolWt(mol)
    logp = Descriptors.MolLogP(mol)
    h_donors = RDKitLipinski.NumHDonors(mol)
    h_acceptors = RDKitLipinski.NumHAcceptors(mol)

    # If it's a peptide, we use "beyond Rule of 5" (bRo5) guidelines
    if is_peptide:
        # Peptides generally target protein-protein interactions (like MDM2)
        # where higher MW and more H-bonds are required for binding surface area.
        violations = 0
        if mw >= 2000:
            violations += 1       # Relaxed for large peptides
        if not (-2 <= logp <= 8):
            violations += 1 # Wider logP tolerance
        return violations <= 1

    # Standard Small Molecule Lipinski
    violations = 0
    if mw >= 500:
        violations += 1
    if logp >= 5:
        violations += 1
    if h_donors >= 5:
        violations += 1
    if h_acceptors >= 10:
        violations += 1
    return violations <= 1


def passes_pains_filter(mol: Chem.Mol) -> bool:
    """Filters out Pan-Assay Interference Compounds (PAINS) using RDKit.

    Instantly drops molecules containing substructures known to cause
    false positives in biological assays.
    """
    from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams

    if mol is None:
        return False

    # Initialize PAINS catalog descriptor
    params = FilterCatalogParams()
    params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS)
    catalog = FilterCatalog(params)

    # If the molecule matches any entry in the PAINS catalog, it fails
    return not catalog.HasMatch(mol)


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
# Chemical, ESM2, Rosetta dSASA, and AF2-PAE filters
# ---------------------------------------------------------------------------

def run_chemical_filter(
    candidate_fasta: str,
) -> FilterResult:
    """Run RDKit chemical validation (Lipinski Rule of 5 and/or PAINS).

    Automatically skips Lipinski check for peptide-like sequences.
    """
    try:
        from rdkit import Chem
        from rdkit.Chem import Descriptors
        from rdkit.Chem import Lipinski as RDKitLipinski
    except ImportError as exc:
        return FilterResult(
            filter_name="Chemical (RDKit)",
            passed=False,
            iptm=0.0,
            threshold=1.0,
            error=f"RDKit is not installed: {exc}",
        )

    try:
        seq = _extract_sequence(candidate_fasta)
        is_peptide = _is_peptide_sequence(seq)

        # Convert FASTA sequence to RDKit Molecule
        mol = Chem.MolFromFASTA(seq)
        if mol is None:
            # Try parsing the whole FASTA
            mol = Chem.MolFromFASTA(candidate_fasta)
            if mol is None:
                return FilterResult(
                    filter_name="Chemical (RDKit)",
                    passed=False,
                    iptm=0.0,
                    threshold=1.0,
                    error="RDKit could not convert the FASTA sequence to a molecule.",
                )

        pains_pass = passes_pains_filter(mol)

        raw_scores = {
            "is_peptide": is_peptide,
            "pains_pass": pains_pass,
        }

        if is_peptide:
            # Skip Lipinski, only PAINS filter determines pass/fail
            passed = pains_pass
            raw_scores["lipinski_pass"] = True
            raw_scores["skipped_lipinski"] = True
        else:
            lipinski_pass = check_lipinski(mol, is_peptide=False)
            passed = pains_pass and lipinski_pass
            raw_scores["lipinski_pass"] = lipinski_pass
            raw_scores["skipped_lipinski"] = False
            raw_scores["mw"] = Descriptors.MolWt(mol)
            raw_scores["logp"] = Descriptors.MolLogP(mol)
            raw_scores["h_donors"] = RDKitLipinski.NumHDonors(mol)
            raw_scores["h_acceptors"] = RDKitLipinski.NumHAcceptors(mol)

        return FilterResult(
            filter_name="Chemical (RDKit)",
            passed=passed,
            iptm=1.0 if passed else 0.0,
            threshold=1.0,
            raw_scores=raw_scores,
        )
    except Exception as exc:
        return FilterResult(
            filter_name="Chemical (RDKit)",
            passed=False,
            iptm=0.0,
            threshold=1.0,
            error=f"Chemical filter execution failed: {exc}",
        )


def run_esm2_ll_filter(
    candidate_fasta: str,
    threshold: float = -2.0,
) -> FilterResult:
    """Run ESM2 log-likelihood expression metric using Hugging Face transformers.

    Computes the mean per-token log-likelihood under the ESM2 model.
    """
    try:
        import torch
        from transformers import AutoTokenizer, AutoModelForMaskedLM
    except ImportError as exc:
        return FilterResult(
            filter_name="ESM2 LL",
            passed=False,
            iptm=0.0,
            threshold=threshold,
            error=f"Transformers or PyTorch not installed: {exc}",
        )

    try:
        seq = _extract_sequence(candidate_fasta)
        if not seq:
            return FilterResult(
                filter_name="ESM2 LL",
                passed=False,
                iptm=0.0,
                threshold=threshold,
                error="Empty candidate sequence",
            )

        tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t6_8M_UR50D")
        model = AutoModelForMaskedLM.from_pretrained("facebook/esm2_t6_8M_UR50D")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)
        model.eval()

        inputs = tokenizer(seq, return_tensors="pt")
        input_ids = inputs["input_ids"].to(device)
        seq_len = input_ids.shape[1]

        num_amino_acids = seq_len - 2
        if num_amino_acids <= 0:
            return FilterResult(
                filter_name="ESM2 LL",
                passed=False,
                iptm=0.0,
                threshold=threshold,
                error="Sequence too short for ESM2 processing",
            )

        mask_token_id = tokenizer.mask_token_id
        batch_input_ids = input_ids.repeat(num_amino_acids, 1)

        for i in range(num_amino_acids):
            batch_input_ids[i, i + 1] = mask_token_id

        with torch.no_grad():
            outputs = model(batch_input_ids)
            logits = outputs.logits

        log_probs = torch.log_softmax(logits, dim=-1)

        total_log_likelihood = 0.0
        for i in range(num_amino_acids):
            true_token_id = input_ids[0, i + 1].item()
            log_prob = log_probs[i, i + 1, true_token_id].item()
            total_log_likelihood += log_prob

        mean_ll = total_log_likelihood / num_amino_acids
        passed = mean_ll >= threshold

        return FilterResult(
            filter_name="ESM2 LL",
            passed=passed,
            iptm=mean_ll,
            threshold=threshold,
            raw_scores={
                "mean_log_likelihood": mean_ll,
                "model_name": "facebook/esm2_t6_8M_UR50D",
                "sequence_length": len(seq),
            }
        )
    except Exception as exc:
        return FilterResult(
            filter_name="ESM2 LL",
            passed=False,
            iptm=0.0,
            threshold=threshold,
            error=f"ESM2 filter execution failed: {exc}",
        )


def run_rosetta_sasa_filter(
    candidate_fasta: str,
    target_fasta: str,
    sasa_threshold: float = 500.0,
) -> FilterResult:
    """Run Rosetta ΔSASA docking filter using PyRosetta.

    Estimates the change in solvent-accessible surface area on complex formation
    via PyRosetta's InterfaceAnalyzerMover.
    """
    try:
        import pyrosetta
        from pyrosetta.rosetta.protocols.analysis import InterfaceAnalyzerMover
    except ImportError as exc:
        return FilterResult(
            filter_name="Rosetta dSASA",
            passed=False,
            iptm=0.0,
            threshold=sasa_threshold,
            error=f"PyRosetta is not installed: {exc}",
        )

    try:
        candidate_seq = _extract_sequence(candidate_fasta)
        target_seq = _extract_sequence(target_fasta)

        if not candidate_seq or not target_seq:
            return FilterResult(
                filter_name="Rosetta dSASA",
                passed=False,
                iptm=0.0,
                threshold=sasa_threshold,
                error="Empty candidate or target sequence",
            )

        # Initialize pyrosetta if not already done
        if not hasattr(pyrosetta, "_initialized") or not pyrosetta._initialized:
            try:
                pyrosetta.init(options="-mute all")
                pyrosetta._initialized = True
            except Exception:
                pyrosetta.init()

        # Build poses for candidate and target
        pose_binder = pyrosetta.pose_from_sequence(candidate_seq)
        pose_target = pyrosetta.pose_from_sequence(target_seq)

        # Append poses to create a two-chain complex (chain 1: binder, chain 2: target)
        pyrosetta.rosetta.core.pose.append_pose_to_pose(pose_binder, pose_target, new_chain=True)

        # Set up InterfaceAnalyzerMover
        # By default, chain 1 is 'A', chain 2 is 'B'
        ia = InterfaceAnalyzerMover("A_B")
        ia.set_pack_separated(True)
        ia.set_pack_input(True)
        ia.apply(pose_binder)

        # Retrieve dSASA value from the pose scores
        dsasa = None
        if hasattr(pose_binder, "scores"):
            try:
                dsasa = pose_binder.scores.get("interface_dSASA")
            except Exception:
                try:
                    dsasa = pose_binder.scores["interface_dSASA"]
                except Exception:
                    pass

        if dsasa is None:
            # Fallback to try accessing it directly from the mover
            for attr in ["get_interface_d_sasa", "interface_d_sasa", "get_interface_dSASA"]:
                if hasattr(ia, attr):
                    try:
                        val = getattr(ia, attr)()
                        dsasa = float(val)
                        break
                    except Exception:
                        pass

        if dsasa is None:
            return FilterResult(
                filter_name="Rosetta dSASA",
                passed=False,
                iptm=0.0,
                threshold=sasa_threshold,
                error="InterfaceAnalyzerMover run succeeded, but interface_dSASA score was not found.",
            )

        passed = dsasa >= sasa_threshold

        return FilterResult(
            filter_name="Rosetta dSASA",
            passed=passed,
            iptm=dsasa,
            threshold=sasa_threshold,
            raw_scores={
                "interface_dsasa": dsasa,
            }
        )
    except Exception as exc:
        return FilterResult(
            filter_name="Rosetta dSASA",
            passed=False,
            iptm=0.0,
            threshold=sasa_threshold,
            error=f"Rosetta filter execution failed: {exc}",
        )


def run_af2_pae_filter(
    candidate_fasta: str,
    target_fasta: str,
    af2ig_output_dir: str | None = None,
    pae_threshold: float = 10.0,
    iptm_threshold: float = 0.6,
) -> FilterResult:
    """Extract inter-chain PAE and iPTM from ColabFold prediction results.

    Reuses the output of AF2-IG if available; otherwise runs AF2-IG.
    """
    candidate_seq = _extract_sequence(candidate_fasta)
    target_seq = _extract_sequence(target_fasta)

    len_candidate = len(candidate_seq)

    # Locate AF2-IG output directory
    output_path = None
    if af2ig_output_dir and os.path.exists(af2ig_output_dir):
        output_path = Path(af2ig_output_dir)
    else:
        # Run AF2-IG to get output directory
        try:
            print("[AF2-PAE] No AF2-IG output directory provided. Running AF2-IG first...")
            af2ig_result = run_af2ig_filter(
                candidate_fasta, target_fasta, iptm_threshold=iptm_threshold
            )
            if af2ig_result.output_dir and os.path.exists(af2ig_result.output_dir):
                output_path = Path(af2ig_result.output_dir)
            else:
                return FilterResult(
                    filter_name="AF2-PAE",
                    passed=False,
                    iptm=0.0,
                    threshold=iptm_threshold,
                    error=f"Could not run AF2-IG to obtain PAE matrix: {af2ig_result.error}",
                )
        except Exception as exc:
            return FilterResult(
                filter_name="AF2-PAE",
                passed=False,
                iptm=0.0,
                threshold=iptm_threshold,
                error=f"Failed running AF2-IG: {exc}",
            )

    score_files = sorted(glob.glob(str(output_path / "*_scores_rank_*.json")))
    if not score_files:
        score_files = sorted(glob.glob(str(output_path / "*scores*.json")))

    if not score_files:
        return FilterResult(
            filter_name="AF2-PAE",
            passed=False,
            iptm=0.0,
            threshold=iptm_threshold,
            error=f"No score JSON files found in {output_path}",
        )

    try:
        with open(score_files[0], encoding="utf-8") as f:
            data = json.load(f)
    except Exception as exc:
        return FilterResult(
            filter_name="AF2-PAE",
            passed=False,
            iptm=0.0,
            threshold=iptm_threshold,
            error=f"Failed to read/parse score file {score_files[0]}: {exc}",
        )

    # Extract iPTM
    iptm = 0.0
    if "iptm" in data:
        iptm = float(data["iptm"])
    elif "iptm+ptm" in data:
        iptm = float(data["iptm+ptm"])

    # Extract PAE matrix
    pae_raw = data.get("pae")
    if pae_raw is None:
        pae_raw = data.get("predicted_aligned_error")

    if pae_raw is None:
        return FilterResult(
            filter_name="AF2-PAE",
            passed=False,
            iptm=iptm,
            threshold=iptm_threshold,
            error=f"No PAE matrix found in {score_files[0]}",
        )

    try:
        # If it's a flat list, reshape it
        if isinstance(pae_raw, list) and len(pae_raw) > 0 and not isinstance(pae_raw[0], list):
            n = len(pae_raw)
            dim = int(math.sqrt(n))
            if dim * dim == n:
                pae_matrix = [pae_raw[i*dim : (i+1)*dim] for i in range(dim)]
            else:
                return FilterResult(
                    filter_name="AF2-PAE",
                    passed=False,
                    iptm=iptm,
                    threshold=iptm_threshold,
                    error=f"PAE array length {n} is not a perfect square",
                )
        else:
            pae_matrix = pae_raw

        rows = len(pae_matrix)
        if rows == 0:
            raise ValueError("PAE matrix is empty")
        cols = len(pae_matrix[0])

        split_idx = len_candidate

        if split_idx >= rows or split_idx >= cols:
            return FilterResult(
                filter_name="AF2-PAE",
                passed=False,
                iptm=iptm,
                threshold=iptm_threshold,
                error=f"PAE matrix size ({rows}x{cols}) is smaller than candidate length ({len_candidate})",
            )

        inter_chain_values = []

        # Extract block 1 (A -> B)
        for r in range(0, split_idx):
            for c in range(split_idx, cols):
                inter_chain_values.append(pae_matrix[r][c])

        # Extract block 2 (B -> A)
        for r in range(split_idx, rows):
            for c in range(0, split_idx):
                inter_chain_values.append(pae_matrix[r][c])

        if not inter_chain_values:
            return FilterResult(
                filter_name="AF2-PAE",
                passed=False,
                iptm=iptm,
                threshold=iptm_threshold,
                error="No inter-chain PAE values extracted",
            )

        mean_inter_pae = sum(inter_chain_values) / len(inter_chain_values)
        min_inter_pae = min(inter_chain_values)

        passed = (mean_inter_pae <= pae_threshold) and (iptm >= iptm_threshold)

        return FilterResult(
            filter_name="AF2-PAE",
            passed=passed,
            iptm=iptm,
            threshold=iptm_threshold,
            output_dir=str(output_path),
            raw_scores={
                "iptm": iptm,
                "mean_interchain_pae": mean_inter_pae,
                "min_interchain_pae": min_inter_pae,
                "pae_threshold": pae_threshold,
                "iptm_threshold": iptm_threshold,
            }
        )
    except Exception as exc:
        return FilterResult(
            filter_name="AF2-PAE",
            passed=False,
            iptm=iptm if 'iptm' in locals() else 0.0,
            threshold=iptm_threshold,
            error=f"AF2-PAE filter execution failed: {exc}",
        )


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def run_structural_validation(
    candidate_fasta: str,
    target_fasta: str,
    *,
    enable_chemical: bool = True,
    enable_esm2: bool = True,
    enable_rosetta: bool = True,
    enable_af2_pae: bool = True,
    enable_af2ig: bool = True,
    enable_protenix: bool = True,
    chemical_threshold: float = 1.0,
    esm2_threshold: float = -2.0,
    rosetta_threshold: float = 500.0,
    af2_pae_threshold: float = 10.0,
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
        enable_chemical: Whether to run Chemical filter.
        enable_esm2: Whether to run ESM2 LL filter.
        enable_rosetta: Whether to run Rosetta dSASA filter.
        enable_af2_pae: Whether to run AF2-PAE filter.
        enable_af2ig: Whether to run AF2-IG filter.
        enable_protenix: Whether to run Protenix filter.
        chemical_threshold: Passed threshold for Chemical validation (not used directly, defaults to 1.0).
        esm2_threshold: Minimum mean LL score for ESM2 filter.
        rosetta_threshold: Minimum dSASA score for Rosetta filter.
        af2_pae_threshold: Maximum mean inter-chain PAE score for AF2-PAE filter.
        af2ig_threshold: ipTM threshold for AF2-IG.
        protenix_threshold: ipTM threshold for Protenix.
        protenix_seeds: Number of Protenix prediction seeds.
    """
    results: list[FilterResult] = []

    # 1. Chemical validation (RDKit)
    if enable_chemical:
        print("\n" + "=" * 60)
        print("STRUCTURAL FILTER: Chemical (RDKit)")
        print("=" * 60)
        try:
            chemical_result = run_chemical_filter(candidate_fasta)
            results.append(chemical_result)
        except Exception as exc:
            print(f"[Chemical] Failed: {exc}")
            results.append(FilterResult(
                filter_name="Chemical (RDKit)",
                passed=False,
                iptm=0.0,
                threshold=1.0,
                error=str(exc),
            ))

    # 2. ESM2 Log-Likelihood
    if enable_esm2:
        print("\n" + "=" * 60)
        print("STRUCTURAL FILTER: ESM2 LL")
        print("=" * 60)
        try:
            esm2_result = run_esm2_ll_filter(candidate_fasta, esm2_threshold)
            results.append(esm2_result)
        except Exception as exc:
            print(f"[ESM2 LL] Failed: {exc}")
            results.append(FilterResult(
                filter_name="ESM2 LL",
                passed=False,
                iptm=0.0,
                threshold=esm2_threshold,
                error=str(exc),
            ))

    # 3. AF2-IG
    af2ig_output_dir = None
    if enable_af2ig:
        print("\n" + "=" * 60)
        print("STRUCTURAL FILTER: AF2-IG (ColabFold)")
        print("=" * 60)
        try:
            af2ig_result = run_af2ig_filter(
                candidate_fasta, target_fasta, af2ig_threshold
            )
            results.append(af2ig_result)
            if af2ig_result.output_dir:
                af2ig_output_dir = af2ig_result.output_dir
        except FileNotFoundError as exc:
            print(f"[AF2-IG] Skipped: {exc}")
            results.append(FilterResult(
                filter_name="AF2-IG",
                passed=False,
                iptm=0.0,
                threshold=af2ig_threshold,
                error=str(exc),
            ))

    # 4. AF2-PAE
    if enable_af2_pae:
        print("\n" + "=" * 60)
        print("STRUCTURAL FILTER: AF2-PAE (PAE + iPTM)")
        print("=" * 60)
        try:
            af2_pae_result = run_af2_pae_filter(
                candidate_fasta,
                target_fasta,
                af2ig_output_dir=af2ig_output_dir,
                pae_threshold=af2_pae_threshold,
                iptm_threshold=af2ig_threshold,
            )
            results.append(af2_pae_result)
        except Exception as exc:
            print(f"[AF2-PAE] Failed: {exc}")
            results.append(FilterResult(
                filter_name="AF2-PAE",
                passed=False,
                iptm=0.0,
                threshold=af2ig_threshold,
                error=str(exc),
            ))

    # 5. Rosetta dSASA
    if enable_rosetta:
        print("\n" + "=" * 60)
        print("STRUCTURAL FILTER: Rosetta dSASA")
        print("=" * 60)
        try:
            rosetta_result = run_rosetta_sasa_filter(
                candidate_fasta, target_fasta, rosetta_threshold
            )
            results.append(rosetta_result)
        except Exception as exc:
            print(f"[Rosetta dSASA] Failed: {exc}")
            results.append(FilterResult(
                filter_name="Rosetta dSASA",
                passed=False,
                iptm=0.0,
                threshold=rosetta_threshold,
                error=str(exc),
            ))

    # 6. Protenix
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
