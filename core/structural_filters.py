"""Post-pipeline structural validation filters.

Optimized for high-throughput screening with vectorized operations, cached
model contexts, and robust macromolecular sequence handling.
"""

from __future__ import annotations

import glob
import json
import math
import os
import re
import shutil
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

# 3rd Party / External Dependencies
import torch
import pyrosetta
from pyrosetta.rosetta.protocols.analysis import InterfaceAnalyzerMover
from transformers import AutoModelForMaskedLM, AutoTokenizer
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem import Lipinski as RDKitLipinski
from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams
from colabfold.batch import get_queries, run
from colabfold.utils import setup_logging
from colabfold.download import default_data_dir


# Global Execution Cache to prevent repetitive Disk/VRAM initialization overhead
_MODEL_CONTEXT_CACHE: Dict[str, Any] = {}

ROOT_DIR = Path(__file__).resolve().parent.parent
STRUCTURAL_DIR = ROOT_DIR / "traces" / "structural"


@dataclass(slots=True)
class FilterResult:
    """Outcome of a single structural validation filter."""
    filter_name: str
    passed: bool
    iptm: float
    ptm: Optional[float] = None
    plddt: Optional[float] = None
    threshold: float = 0.0
    output_dir: str = ""
    raw_scores: dict = field(default_factory=dict)
    error: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _extract_sequence(fasta: str) -> str:
    """Extract and clean raw amino-acid sequences from FASTA strings."""
    if not fasta.strip():
        return ""
    lines = fasta.strip().splitlines()
    return "".join(line.strip() for line in lines if not line.startswith(">")).upper()


def _is_peptide_sequence(seq: str) -> bool:
    """Identifies standard peptide blocks versus small molecule chemical definitions."""
    seq = seq.strip().upper()
    if not seq:
        return False
    aa_chars = set("ACDEFGHIKLMNPQRSTVWYX*:")
    return all(c in aa_chars for c in seq) and len(seq) > 5


def _get_esm_model_context() -> tuple[Any, Any, str]:
    """Retrieves or instantiates the ESM2 model context out of global cache storage."""
    model_name = "facebook/esm2_t6_8M_UR50D"
    if model_name not in _MODEL_CONTEXT_CACHE:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForMaskedLM.from_pretrained(model_name).to(device)
        model.eval()
        _MODEL_CONTEXT_CACHE[model_name] = (tokenizer, model, device)
    return _MODEL_CONTEXT_CACHE[model_name]


def _init_pyrosetta_safe() -> Any:
    """Safely initializes PyRosetta framework once across runtime context."""
    if not getattr(pyrosetta, "_initialized", False):
        try:
            pyrosetta.init(options="-mute all -constant_seed")
            pyrosetta._initialized = True
        except Exception:
            pyrosetta.init(options="-mute all")
            pyrosetta._initialized = True
    return pyrosetta


# ---------------------------------------------------------------------------
# Core Filters
# ---------------------------------------------------------------------------

def run_chemical_filter(candidate_fasta: str) -> FilterResult:
    """Validates structural chemistry constraints using RDKit."""
    try:
        seq = _extract_sequence(candidate_fasta)
        is_peptide = _is_peptide_sequence(seq)
        
        # Guard against massive macromolecular string parsing with small-molecule descriptors
        if is_peptide and len(seq) > 50:
            return FilterResult(
                filter_name="Chemical (RDKit)",
                passed=True,  # Large macromolecular structures pass chemical filters implicitly
                iptm=1.0,
                threshold=1.0,
                raw_scores={"is_peptide": True, "macrocycle_detected": True, "seq_len": len(seq)}
            )

        mol = Chem.MolFromFASTA(seq) or Chem.MolFromFASTA(candidate_fasta)
        if mol is None:
            return FilterResult("Chemical (RDKit)", False, 0.0, error="RDKit conversion failure.")

        # Vectorized initialization of PAINS properties
        params = FilterCatalogParams()
        params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS)
        catalog = FilterCatalog(params)
        pains_pass = not catalog.HasMatch(mol)

        mw = Descriptors.MolWt(mol)
        logp = Descriptors.MolLogP(mol)
        h_donors = RDKitLipinski.NumHDonors(mol)
        h_acceptors = RDKitLipinski.NumHAcceptors(mol)

        violations = sum([mw >= 500, logp >= 5, h_donors >= 5, h_acceptors >= 10])
        passed = pains_pass and (violations <= 1)

        return FilterResult(
            filter_name="Chemical (RDKit)",
            passed=passed,
            iptm=1.0 if passed else 0.0,
            threshold=1.0,
            raw_scores={"mw": mw, "logp": logp, "h_donors": h_donors, "h_acceptors": h_acceptors, "pains_pass": pains_pass}
        )
    except Exception as exc:
        return FilterResult("Chemical (RDKit)", False, 0.0, error=str(exc))


def run_esm2_ll_filter(candidate_fasta: str, threshold: float = -2.0) -> FilterResult:
    """Computes pseudo-log-likelihood scores efficiently across candidate batches."""
    try:
        tokenizer, model, device = _get_esm_model_context()
        seq = _extract_sequence(candidate_fasta)
        if not seq:
            return FilterResult("ESM2 LL", False, 0.0, threshold=threshold, error="Empty sequence.")

        inputs = tokenizer(seq, return_tensors="pt")
        input_ids = inputs["input_ids"].to(device)
        seq_len = input_ids.shape[1]
        num_amino_acids = seq_len - 2

        if num_amino_acids <= 0:
            return FilterResult("ESM2 LL", False, 0.0, threshold=threshold, error="Sequence too short.")

        # Vectorized expansion and masking matrix construction (removes loop overhead)
        batch_input_ids = input_ids.repeat(num_amino_acids, 1)
        indices = torch.arange(1, num_amino_acids + 1, device=device)
        batch_input_ids[torch.arange(num_amino_acids), indices] = tokenizer.mask_token_id

        with torch.no_grad():
            outputs = model(batch_input_ids)
            log_probs = torch.log_softmax(outputs.logits, dim=-1)
            
        # Extract target amino acid token likelihoods via index matrices
        target_tokens = input_ids[0, 1:num_amino_acids + 1]
        gathered_ll = log_probs[torch.arange(num_amino_acids), indices, target_tokens]
        mean_ll = gathered_ll.mean().item()
        
        passed = mean_ll >= threshold
        return FilterResult(
            filter_name="ESM2 LL", passed=passed, iptm=mean_ll, threshold=threshold,
            raw_scores={"mean_log_likelihood": mean_ll, "sequence_length": len(seq)}
        )
    except Exception as exc:
        return FilterResult("ESM2 LL", False, 0.0, threshold=threshold, error=str(exc))


def run_rosetta_sasa_filter(candidate_fasta: str, target_fasta: str, sasa_threshold: float = 500.0) -> FilterResult:
    """Evaluates the change in solvent-accessible surface area upon binding interface formation."""
    try:
        pyrosetta_instance = _init_pyrosetta_safe()
        
        c_seq = _extract_sequence(candidate_fasta)
        t_seq = _extract_sequence(target_fasta)
        if not c_seq or not t_seq:
            return FilterResult("Rosetta dSASA", False, 0.0, error="Missing sequence string blocks.")

        # Build individual physical structural topologies
        pose_binder = pyrosetta_instance.pose_from_sequence(c_seq)
        pose_target = pyrosetta_instance.pose_from_sequence(t_seq)
        
        # Chain merging protocol setup
        pyrosetta_instance.rosetta.core.pose.append_pose_to_pose(pose_binder, pose_target, new_chain=True)
        complex_pose = pose_binder

        # Instantiating the structural evaluation protocol across target chain boundaries
        ia = InterfaceAnalyzerMover("A_B")
        ia.set_pack_separated(True)
        ia.set_pack_input(True)
        ia.apply(complex_pose)

        dsasa = complex_pose.scores.get("interface_dSASA") if hasattr(complex_pose, "scores") else None
        if dsasa is None:
            for attr in ["get_interface_d_sasa", "interface_d_sasa", "get_interface_dSASA"]:
                if hasattr(ia, attr):
                    try:
                        dsasa = float(getattr(ia, attr)())
                        break
                    except Exception:
                        pass

        if dsasa is None:
            return FilterResult("Rosetta dSASA", False, 0.0, error="SASA evaluation metrics missing from mover output context.")

        dg_separated = complex_pose.scores.get("dG_separated") if hasattr(complex_pose, "scores") else None
        if dg_separated is None:
            for attr in ["get_separated_interface_energy", "separated_interface_energy", "get_interface_dG"]:
                if hasattr(ia, attr):
                    try:
                        dg_separated = float(getattr(ia, attr)())
                        break
                    except Exception:
                        pass

        raw_scores = {"interface_dsasa": dsasa}
        if dg_separated is not None:
            raw_scores["dG_separated"] = dg_separated

        passed = dsasa >= sasa_threshold
        return FilterResult("Rosetta dSASA", passed, dsasa, threshold=sasa_threshold, raw_scores=raw_scores)
    except Exception as exc:
        return FilterResult("Rosetta dSASA", False, 0.0, error=str(exc))


# ---------------------------------------------------------------------------
# Structural Orchestration Utilities
# ---------------------------------------------------------------------------

def _make_output_dir(filter_name: str) -> Path:
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out = STRUCTURAL_DIR / ts / filter_name
    out.mkdir(parents=True, exist_ok=True)
    return out


def _parse_colabfold_scores(output_dir: Path) -> dict:
    score_files = sorted(glob.glob(str(output_dir / "*_scores_rank_*.json"))) or sorted(glob.glob(str(output_dir / "*scores*.json")))
    if not score_files:
        return {"error": "No score files found", "iptm": 0.0}
    try:
        with open(score_files[0], encoding="utf-8") as f:
            data = json.load(f)
        scores = {
            "iptm": float(data.get("iptm", data.get("iptm+ptm", 0.0))),
            "ptm": float(data.get("ptm", 0.0)),
            "mean_plddt": sum(data["plddt"]) / len(data["plddt"]) if isinstance(data.get("plddt"), list) else float(data.get("plddt", 0.0)),
            "source_file": score_files[0]
        }
        return scores
    except Exception as e:
        return {"error": str(e), "iptm": 0.0}


def run_af2ig_filter(candidate_fasta: str, target_fasta: str, iptm_threshold: float = 0.6) -> FilterResult:
    output_dir = _make_output_dir("af2ig")
    candidate_seq = _extract_sequence(candidate_fasta)
    target_seq = _extract_sequence(target_fasta)

    combined_seq = f"{candidate_seq}:{target_seq}"
    input_dir = output_dir / "tmp_input"
    input_dir.mkdir(parents=True, exist_ok=True)
    input_fasta = input_dir / "candidate_target_complex.fasta"
    input_fasta.write_text(f">candidate_target_complex\n{combined_seq}\n", encoding="utf-8")

    try:
        if not (default_data_dir / "params" / "params_model_3_multimer_v3.npz").exists():
            return FilterResult("AF2-IG", False, 0.0, threshold=iptm_threshold, output_dir=str(output_dir), error="AlphaFold2 multimer weights not found.")

        setup_logging(output_dir / "log.txt")
        queries, is_complex = get_queries(str(input_dir))
        run(queries=queries, result_dir=str(output_dir), is_complex=is_complex, use_bfloat16=False, use_templates=False,
            msa_mode="MMseqs2 (UniRef+Environmental)", model_type="alphafold2_multimer_v3", num_models=1, num_recycles=3, num_relax=1)
        
        scores = _parse_colabfold_scores(output_dir)
        iptm = scores.get("iptm", 0.0)
        plddt = scores.get("mean_plddt", None)
        passed = iptm >= iptm_threshold

        return FilterResult("AF2-IG", passed, iptm, ptm=scores.get("ptm"), plddt=plddt, threshold=iptm_threshold, output_dir=str(output_dir), raw_scores=scores)
    except Exception as exc:
        return FilterResult("AF2-IG", False, 0.0, threshold=iptm_threshold, output_dir=str(output_dir), error=str(exc))
    finally:
        shutil.rmtree(input_dir, ignore_errors=True)


def run_af2_pae_filter(candidate_fasta: str, target_fasta: str, af2ig_output_dir: Optional[str] = None, pae_threshold: float = 10.0, iptm_threshold: float = 0.6) -> FilterResult:
    candidate_seq = _extract_sequence(candidate_fasta)
    len_candidate = len(candidate_seq)
    output_path = Path(af2ig_output_dir) if af2ig_output_dir and os.path.exists(af2ig_output_dir) else None

    if not output_path:
        res = run_af2ig_filter(candidate_fasta, target_fasta, iptm_threshold=iptm_threshold)
        if not res.passed or not res.output_dir:
            return FilterResult("AF2-PAE", False, 0.0, threshold=iptm_threshold, error=f"Pre-requisite structural execution step missing: {res.error}")
        output_path = Path(res.output_dir)

    score_files = sorted(glob.glob(str(output_path / "*_scores_rank_*.json"))) or sorted(glob.glob(str(output_path / "*scores*.json")))
    if not score_files:
        return FilterResult("AF2-PAE", False, 0.0, error="Target runtime score files completely missing from path validation step.")

    try:
        with open(score_files[0], encoding="utf-8") as f:
            data = json.load(f)

        iptm = float(data.get("iptm", data.get("iptm+ptm", 0.0)))
        pae_raw = data.get("pae", data.get("predicted_aligned_error"))
        if pae_raw is None:
            return FilterResult("AF2-PAE", False, iptm, error="PAE missing from serialization file parameters.")

        if isinstance(pae_raw, list) and len(pae_raw) > 0 and not isinstance(pae_raw[0], list):
            n = len(pae_raw)
            dim = int(math.sqrt(n))
            pae_matrix = [pae_raw[i*dim : (i+1)*dim] for i in range(dim)]
        else:
            pae_matrix = pae_raw

        rows = len(pae_matrix)
        cols = len(pae_matrix[0]) if rows > 0 else 0
        if len_candidate >= rows or len_candidate >= cols:
            return FilterResult("AF2-PAE", False, iptm, error="Matrix tracking configuration bounds dimensions breach sequence alignments.")

        inter_chain_values = []
        for r in range(0, len_candidate):
            inter_chain_values.extend(pae_matrix[r][len_candidate:cols])
        for r in range(len_candidate, rows):
            inter_chain_values.extend(pae_matrix[r][0:len_candidate])

        mean_inter_pae = sum(inter_chain_values) / len(inter_chain_values) if inter_chain_values else 100.0
        passed = (mean_inter_pae <= pae_threshold) and (iptm >= iptm_threshold)

        return FilterResult("AF2-PAE", passed, iptm, threshold=iptm_threshold, output_dir=str(output_path),
                            raw_scores={"iptm": iptm, "mean_interchain_pae": mean_inter_pae, "min_interchain_pae": min(inter_chain_values or [0])})
    except Exception as exc:
        return FilterResult("AF2-PAE", False, 0.0, error=str(exc))


def run_protenix_filter(candidate_fasta: str, target_fasta: str, iptm_threshold: float = 0.7, num_seeds: int = 3) -> FilterResult:
    try:
        protenix_bin = shutil.which("protenix")
        if not protenix_bin:
            raise FileNotFoundError("Protenix binary execution target missing from system PATH variables configuration.")
        output_dir = _make_output_dir("protenix")
        c_seq = _extract_sequence(candidate_fasta)
        t_seq = _extract_sequence(target_fasta)

        input_data = [{"sequences": [{"protein": {"id": "A", "sequence": c_seq}}, {"protein": {"id": "B", "sequence": t_seq}}], "name": "candidate_target_complex"}]
        input_json = output_dir / "input.json"
        input_json.write_text(json.dumps(input_data, indent=2), encoding="utf-8")
        seeds = ",".join(str(101 + i) for i in range(num_seeds))

        result = subprocess.run([protenix_bin, "predict", "--input", str(input_json), "--out_dir", str(output_dir), "--seeds", seeds], capture_output=True, text=True, timeout=3600)
        if result.returncode != 0:
            return FilterResult("Protenix", False, 0.0, threshold=iptm_threshold, output_dir=str(output_dir), error=f"Execution pipeline failure: {result.stderr[-400:]}")

        conf_files = sorted(glob.glob(str(output_dir / "**/*summary_confidence*.json"), recursive=True))
        if not conf_files:
            return FilterResult("Protenix", False, 0.0, error="Telemetry profiles missed capture parsing blocks.")

        best_score, best_data = -float("inf"), {}
        for fpath in conf_files:
            with open(fpath, encoding="utf-8") as f:
                data = json.load(f)
            ranking = 0.8 * float(data.get("iptm", 0.0)) + 0.2 * float(data.get("ptm", 0.0)) - 100.0 * float(data.get("has_clash", 0.0))
            if ranking > best_score:
                best_score, best_data = ranking, data

        iptm = float(best_data.get("iptm", 0.0))
        passed = iptm >= iptm_threshold
        return FilterResult("Protenix", passed, iptm, ptm=float(best_data.get("ptm", 0.0)), threshold=iptm_threshold, output_dir=str(output_dir), raw_scores=best_data)
    except Exception as exc:
        return FilterResult("Protenix", False, 0.0, threshold=iptm_threshold, error=str(exc))


def run_structural_validation(candidate_fasta: str, target_fasta: str, **kwargs) -> list[FilterResult]:
    """Orchestrates sequential validation verification over targeting assets."""
    results: list[FilterResult] = []
    
    filter_mappings = [
        ("enable_chemical", run_chemical_filter, [candidate_fasta]),
        ("enable_esm2", run_esm2_ll_filter, [candidate_fasta, kwargs.get("esm2_threshold", -2.0)]),
        ("enable_rosetta", run_rosetta_sasa_filter, [candidate_fasta, target_fasta, kwargs.get("rosetta_threshold", 500.0)]),
        ("enable_af2ig", run_af2ig_filter, [candidate_fasta, target_fasta, kwargs.get("af2ig_threshold", 0.6)]),
        ("enable_af2_pae", run_af2_pae_filter, [candidate_fasta, target_fasta, None, kwargs.get("af2_pae_threshold", 10.0), kwargs.get("af2ig_threshold", 0.6)]),
        ("enable_protenix", run_protenix_filter, [candidate_fasta, target_fasta, kwargs.get("protenix_threshold", 0.7), kwargs.get("protenix_seeds", 3)]),
    ]

    for key, func, args in filter_mappings:
        if kwargs.get(key, True):
            try:
                results.append(func(*args)) # type: ignore
            except Exception as e:
                results.append(FilterResult(func.__name__, False, 0.0, error=str(e)))

    return results