"""Post-pipeline structural validation filters.

Runs AF2-IG (via ColabFold) and/or Protenix to predict whether a designed
candidate binder will actually fold and bind to the target protein.  Both
tools are invoked via subprocess to avoid dependency conflicts between
JAX (ColabFold) and PyTorch (Protenix).

Additional filters:
  - Chemical verification (RDKit Lipinski Ro5 + PAINS): applicable to
    small molecules; automatically skips Lipinski for peptide-like sequences.
  - ESM2 log-likelihood expression metric: mean per-token log-likelihood
    under the ESM2 protein language model, correlated with expression.
  - Rosetta ΔSASA docking: change in solvent-accessible surface area on
    complex formation, estimated via PyRosetta InterfaceAnalyzerMover.
  - AlphaFold2 Multimer PAE interaction + iPTM: inter-chain PAE block
    extracted from the ColabFold score JSON (reuses AF2-IG output).
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

    filter_name: str  # e.g. "AF2-IG", "Protenix", "Chemical-RDKit", …
    passed: bool
    iptm: float       # primary continuous score (repurposed for non-iptm filters)
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
# Filter 1 – Chemical verification (RDKit: Lipinski Ro5 + PAINS)
# ---------------------------------------------------------------------------

def _is_peptide_like_mol(mol: Any) -> bool:
    """Heuristically classify an RDKit molecule as peptide-like.

    Mirrors the logic in smolagent_tools so this module stays self-contained.
    """
    from rdkit import Chem  # local import – optional dependency
    if mol is None:
        return False
    amide_pattern = Chem.MolFromSmarts("C(=O)N")
    amide_count = len(mol.GetSubstructMatches(amide_pattern))
    return amide_count >= 4 and mol.GetNumAtoms() >= 30


def run_chemical_filter(
    candidate_fasta: str,
    *,
    skip_lipinski_for_peptides: bool = True,
) -> FilterResult:
    """Check a candidate against Lipinski Rule of 5 and the PAINS catalog.

    For peptide-like sequences (≥4 amide bonds and ≥30 heavy atoms), the
    Lipinski Ro5 check is automatically skipped because the rule was designed
    for small-molecule oral drugs and does not apply to peptides.  The PAINS
    filter is still run in all cases.

    Args:
        candidate_fasta: Candidate binder in FASTA format.
        skip_lipinski_for_peptides: When True (default) Lipinski is not
            applied to peptide-like molecules.

    Returns:
        FilterResult with ``filter_name="Chemical-RDKit"``.  The ``iptm``
        field holds 1.0 on pass and 0.0 on fail (binary metric).
    """
    try:
        from rdkit import Chem
        from rdkit.Chem import Descriptors
        from rdkit.Chem import Lipinski as RDKitLipinski
        from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams
    except ImportError as exc:
        return FilterResult(
            filter_name="Chemical-RDKit",
            passed=False,
            iptm=0.0,
            error=f"RDKit not available: {exc}",
        )

    print("[Chemical-RDKit] Converting FASTA to molecule…")

    mol = Chem.MolFromFASTA(candidate_fasta)
    if mol is None:
        return FilterResult(
            filter_name="Chemical-RDKit",
            passed=False,
            iptm=0.0,
            error="RDKit could not parse the FASTA sequence into a molecule.",
        )

    peptide_mode = _is_peptide_like_mol(mol) if skip_lipinski_for_peptides else False

    mw = Descriptors.MolWt(mol)
    logp = Descriptors.MolLogP(mol)
    h_donors = RDKitLipinski.NumHDonors(mol)
    h_acceptors = RDKitLipinski.NumHAcceptors(mol)

    lipinski_passed: bool | None = None
    lipinski_violations = 0
    lipinski_reason = ""

    if peptide_mode:
        lipinski_passed = None  # not applicable
        lipinski_reason = "Lipinski Ro5 skipped (peptide-like molecule)"
        print(f"[Chemical-RDKit] {lipinski_reason}")
    else:
        if mw >= 500:
            lipinski_violations += 1
        if logp >= 5:
            lipinski_violations += 1
        if h_donors >= 5:
            lipinski_violations += 1
        if h_acceptors >= 10:
            lipinski_violations += 1
        lipinski_passed = lipinski_violations <= 1
        lipinski_reason = (
            "Passed Lipinski Ro5" if lipinski_passed
            else f"Failed Lipinski Ro5 ({lipinski_violations} violations)"
        )
        print(f"[Chemical-RDKit] Lipinski: {lipinski_reason}")

    # PAINS filter – always run
    params = FilterCatalogParams()
    params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS)
    catalog = FilterCatalog(params)
    pains_passed = not catalog.HasMatch(mol)
    pains_reason = "Passed PAINS filter" if pains_passed else "Flagged by PAINS catalog"
    print(f"[Chemical-RDKit] PAINS: {pains_reason}")

    # Aggregate pass/fail
    if peptide_mode:
        passed = pains_passed          # only PAINS applicable for peptides
    else:
        passed = bool(lipinski_passed) and pains_passed

    raw = {
        "is_peptide": peptide_mode,
        "mw": round(mw, 2),
        "logp": round(logp, 2),
        "h_donors": h_donors,
        "h_acceptors": h_acceptors,
        "lipinski_violations": lipinski_violations if not peptide_mode else None,
        "lipinski_passed": lipinski_passed,
        "lipinski_reason": lipinski_reason,
        "pains_passed": pains_passed,
        "pains_reason": pains_reason,
    }

    print(f"[Chemical-RDKit] Result: {'PASS ✅' if passed else 'FAIL ❌'}")
    return FilterResult(
        filter_name="Chemical-RDKit",
        passed=passed,
        iptm=1.0 if passed else 0.0,
        threshold=0.5,
        raw_scores=raw,
    )


# ---------------------------------------------------------------------------
# Filter 2 – ESM2 log-likelihood expression metric
# ---------------------------------------------------------------------------

# Lazy-loaded ESM2 model cache to avoid repeated loading across calls.
_esm2_model_cache: dict[str, Any] = {}


def _load_esm2(model_name: str) -> tuple[Any, Any, Any]:
    """Load and cache an ESM2 model + tokenizer from Hugging Face.

    Returns:
        (model, tokenizer, device)
    """
    if model_name in _esm2_model_cache:
        return _esm2_model_cache[model_name]

    import torch
    from transformers import AutoTokenizer, EsmForMaskedLM

    print(f"[ESM2-LogLikelihood] Loading model '{model_name}' from Hugging Face…")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = EsmForMaskedLM.from_pretrained(model_name)
    model.eval()

    # Use GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    _esm2_model_cache[model_name] = (model, tokenizer, device)
    return model, tokenizer, device


def run_esm2_ll_filter(
    candidate_fasta: str,
    *,
    model_name: str = "facebook/esm2_t6_8M_UR50D",
    ll_threshold: float = -1.5,
) -> FilterResult:
    """Compute ESM2 mean per-token log-likelihood as an expression proxy.

    Uses Hugging Face ``transformers.EsmForMaskedLM`` to load the ESM2
    protein language model.  A higher (less negative) mean log-likelihood
    indicates that the sequence is more "natural", which correlates with
    expression level in E. coli and mammalian cell systems.

    Args:
        candidate_fasta: Candidate binder in FASTA format.
        model_name: Hugging Face model identifier
            (default: ``facebook/esm2_t6_8M_UR50D``).
        ll_threshold: Minimum mean log-likelihood to pass (default: ``-1.5``).

    Returns:
        FilterResult with ``filter_name="ESM2-LogLikelihood"``.  The ``iptm``
        field holds the mean log-likelihood (primary score).
    """
    try:
        import torch  # noqa: F401 – check availability
        from transformers import AutoTokenizer, EsmForMaskedLM  # noqa: F401
    except ImportError as exc:
        return FilterResult(
            filter_name="ESM2-LogLikelihood",
            passed=False,
            iptm=0.0,
            threshold=ll_threshold,
            error=(
                f"transformers or torch not available: {exc}. "
                "Install with: pip install transformers torch"
            ),
        )

    sequence = _extract_sequence(candidate_fasta)
    if not sequence:
        return FilterResult(
            filter_name="ESM2-LogLikelihood",
            passed=False,
            iptm=0.0,
            threshold=ll_threshold,
            error="Empty sequence extracted from FASTA.",
        )

    print(f"[ESM2-LogLikelihood] Sequence length: {len(sequence)}")
    print(f"[ESM2-LogLikelihood] Model: {model_name}")

    try:
        model, tokenizer, device = _load_esm2(model_name)
    except Exception as exc:
        return FilterResult(
            filter_name="ESM2-LogLikelihood",
            passed=False,
            iptm=0.0,
            threshold=ll_threshold,
            error=f"Failed to load ESM2 model '{model_name}': {exc}",
        )

    try:
        import torch
        import torch.nn.functional as F

        inputs = tokenizer(sequence, return_tensors="pt").to(device)
        input_ids = inputs["input_ids"][0]  # (L+2,)  [CLS ... EOS]

        with torch.no_grad():
            logits = model(**inputs).logits  # (1, L+2, V)

        log_probs = F.log_softmax(logits[0], dim=-1)  # (L+2, V)

        # Positions 1..L are the actual residue tokens (0=CLS, L+1=EOS).
        token_ids = input_ids[1: len(sequence) + 1]  # (L,)
        per_token_ll = log_probs[1: len(sequence) + 1, :].gather(
            dim=-1, index=token_ids.unsqueeze(-1)
        ).squeeze(-1)  # (L,)

        mean_ll = float(per_token_ll.mean().item())
        per_residue_ll = per_token_ll.cpu().tolist()

    except Exception as exc:
        return FilterResult(
            filter_name="ESM2-LogLikelihood",
            passed=False,
            iptm=0.0,
            threshold=ll_threshold,
            error=f"ESM2 inference failed: {exc}",
        )

    passed = mean_ll >= ll_threshold

    print(f"[ESM2-LogLikelihood] Mean log-likelihood: {mean_ll:.4f} (threshold: {ll_threshold})")
    print(f"[ESM2-LogLikelihood] Result: {'PASS ✅' if passed else 'FAIL ❌'}")

    return FilterResult(
        filter_name="ESM2-LogLikelihood",
        passed=passed,
        iptm=mean_ll,
        threshold=ll_threshold,
        raw_scores={
            "mean_log_likelihood": mean_ll,
            "model": model_name,
            "sequence_length": len(sequence),
            "per_residue_ll": per_residue_ll,
        },
    )


# ---------------------------------------------------------------------------
# Filter 3 – Rosetta ΔSASA docking
# ---------------------------------------------------------------------------

def run_rosetta_sasa_filter(
    candidate_fasta: str,
    target_fasta: str,
    *,
    sasa_threshold: float = 500.0,
) -> FilterResult:
    """Estimate binding interface area via Rosetta InterfaceAnalyzerMover (ΔSASA).

    Builds a simple two-chain complex from the candidate and target sequences,
    relaxes side chains, and measures how much SASA is buried at the interface.
    A larger ΔSASA indicates a more compact, well-packed interface.

    Requires PyRosetta (free academic licence):
        pip install pyrosetta-installer
        python -c "import pyrosetta_installer; pyrosetta_installer.install_pyrosetta()"

    Args:
        candidate_fasta: Candidate binder in FASTA format.
        target_fasta: Target protein in FASTA format.
        sasa_threshold: Minimum ΔSASA (Å²) to pass (default: 500 Å²).

    Returns:
        FilterResult with ``filter_name="Rosetta-SASA"``.  The ``iptm``
        field holds the computed ΔSASA value.
    """
    try:
        import pyrosetta
        from pyrosetta import pose_from_sequence
        from pyrosetta.rosetta.protocols.analysis import InterfaceAnalyzerMover
    except ImportError as exc:
        return FilterResult(
            filter_name="Rosetta-SASA",
            passed=False,
            iptm=0.0,
            threshold=sasa_threshold,
            error=(
                f"PyRosetta not available: {exc}. "
                "Install with: pip install pyrosetta-installer && "
                "python -c 'import pyrosetta_installer; pyrosetta_installer.install_pyrosetta()'"
            ),
        )

    candidate_seq = _extract_sequence(candidate_fasta)
    target_seq = _extract_sequence(target_fasta)

    if not candidate_seq or not target_seq:
        return FilterResult(
            filter_name="Rosetta-SASA",
            passed=False,
            iptm=0.0,
            threshold=sasa_threshold,
            error="Could not extract sequences from provided FASTA strings.",
        )

    print(f"[Rosetta-SASA] Initialising PyRosetta…")
    print(f"[Rosetta-SASA] Candidate length: {len(candidate_seq)}")
    print(f"[Rosetta-SASA] Target length: {len(target_seq)}")

    try:
        # Suppress verbose Rosetta output
        pyrosetta.init(
            "-mute all -ex1 -ex2aro",
            silent=True,
        )

        # Build individual poses from sequence
        pose_candidate = pose_from_sequence(candidate_seq, "fa_standard")
        pose_target = pose_from_sequence(target_seq, "fa_standard")

        # Measure SASA of each monomer (free state)
        from pyrosetta.rosetta.core.scoring.sasa import SasaCalc
        sasa_calc = SasaCalc()

        sasa_calc.calculate(pose_candidate)
        sasa_free_candidate = sasa_calc.get_total_sasa()

        sasa_calc.calculate(pose_target)
        sasa_free_target = sasa_calc.get_total_sasa()

        # Assemble complex (chain A = candidate, chain B = target)
        from pyrosetta.rosetta.core.pose import append_pose_to_pose
        complex_pose = pose_candidate.clone()
        append_pose_to_pose(complex_pose, pose_target, new_chain=True)

        # Quick side-chain repack at interface
        from pyrosetta.rosetta.core.scoring import get_score_function
        from pyrosetta.rosetta.protocols.minimization_packing import PackRotamersMover
        from pyrosetta.rosetta.core.pack.task import TaskFactory
        from pyrosetta.rosetta.core.pack.task.operation import RestrictToRepacking

        sfxn = get_score_function()
        tf = TaskFactory()
        tf.push_back(RestrictToRepacking())
        packer = PackRotamersMover(sfxn, tf.create_task_and_apply_taskoperations(complex_pose))
        packer.apply(complex_pose)

        # Measure complex SASA
        sasa_calc.calculate(complex_pose)
        sasa_complex = sasa_calc.get_total_sasa()

        delta_sasa = (sasa_free_candidate + sasa_free_target) - sasa_complex
        passed = delta_sasa >= sasa_threshold

        print(f"[Rosetta-SASA] SASA candidate (free): {sasa_free_candidate:.1f} Å²")
        print(f"[Rosetta-SASA] SASA target (free):    {sasa_free_target:.1f} Å²")
        print(f"[Rosetta-SASA] SASA complex:          {sasa_complex:.1f} Å²")
        print(f"[Rosetta-SASA] ΔSASA: {delta_sasa:.1f} Å² (threshold: {sasa_threshold} Å²)")
        print(f"[Rosetta-SASA] Result: {'PASS ✅' if passed else 'FAIL ❌'}")

        return FilterResult(
            filter_name="Rosetta-SASA",
            passed=passed,
            iptm=delta_sasa,
            threshold=sasa_threshold,
            raw_scores={
                "sasa_free_candidate": round(sasa_free_candidate, 2),
                "sasa_free_target": round(sasa_free_target, 2),
                "sasa_complex": round(sasa_complex, 2),
                "delta_sasa": round(delta_sasa, 2),
            },
        )

    except Exception as exc:
        return FilterResult(
            filter_name="Rosetta-SASA",
            passed=False,
            iptm=0.0,
            threshold=sasa_threshold,
            error=f"PyRosetta SASA calculation failed: {exc}",
        )


# ---------------------------------------------------------------------------
# Filter 4 – AlphaFold2 Multimer PAE interaction + iPTM
# ---------------------------------------------------------------------------

def _parse_af2_pae(
    output_dir: Path,
    candidate_len: int,
) -> dict[str, Any]:
    """Extract inter-chain PAE and iPTM from ColabFold multimer score JSON.

    ColabFold writes ``*_scores_rank_001_*.json`` (or similar).  The JSON
    contains a ``pae`` field that is a flattened (L×L) matrix of predicted
    aligned errors in Ångströms.  We extract the off-diagonal blocks that
    correspond to cross-chain pairs (candidate→target and target→candidate)
    and compute the mean as the inter-chain PAE score.

    Args:
        output_dir: Path to the ColabFold output directory.
        candidate_len: Length of the candidate chain (chain A).

    Returns:
        dict with keys ``iptm``, ``ptm``, ``mean_inter_pae``, ``pae_matrix``,
        ``source_file``.
    """
    score_files = sorted(glob.glob(str(output_dir / "*_scores_rank_*.json")))
    if not score_files:
        score_files = sorted(glob.glob(str(output_dir / "*scores*.json")))

    if not score_files:
        return {"error": "No score files found", "iptm": 0.0, "mean_inter_pae": 999.0}

    with open(score_files[0], encoding="utf-8") as f:
        data = json.load(f)

    result: dict[str, Any] = {"source_file": score_files[0]}

    # iPTM
    result["iptm"] = float(data.get("iptm", data.get("iptm+ptm", 0.0)))
    if "ptm" in data:
        result["ptm"] = float(data["ptm"])

    # PAE matrix – may be stored as flat list (L*L) or list-of-lists
    pae_raw = data.get("pae") or data.get("predicted_aligned_error")
    if pae_raw is None:
        result["mean_inter_pae"] = None
        result["pae_note"] = "PAE matrix not present in score file"
        return result

    # Normalise to list-of-lists
    if isinstance(pae_raw, list) and pae_raw and isinstance(pae_raw[0], list):
        pae_matrix = pae_raw  # already L×L
    elif isinstance(pae_raw, list):
        # Flat list → infer L
        total = len(pae_raw)
        L = int(total ** 0.5)
        if L * L != total:
            result["mean_inter_pae"] = None
            result["pae_note"] = f"PAE flat list length {total} is not a perfect square"
            return result
        pae_matrix = [pae_raw[i * L:(i + 1) * L] for i in range(L)]
    else:
        result["mean_inter_pae"] = None
        result["pae_note"] = "Unrecognised PAE format"
        return result

    L_total = len(pae_matrix)
    L_a = candidate_len          # chain A (candidate) residues
    L_b = L_total - L_a          # chain B (target) residues

    if L_a <= 0 or L_b <= 0:
        result["mean_inter_pae"] = None
        result["pae_note"] = (
            f"Unexpected chain lengths: A={L_a}, B={L_b}, total={L_total}"
        )
        return result

    # Off-diagonal inter-chain blocks:
    #   top-right  : pae[i][j] for i in [0, L_a), j in [L_a, L_total)
    #   bottom-left: pae[i][j] for i in [L_a, L_total), j in [0, L_a)
    inter_vals: list[float] = []
    for i in range(L_a):
        for j in range(L_a, L_total):
            inter_vals.append(float(pae_matrix[i][j]))
    for i in range(L_a, L_total):
        for j in range(L_a):
            inter_vals.append(float(pae_matrix[i][j]))

    result["mean_inter_pae"] = (
        sum(inter_vals) / len(inter_vals) if inter_vals else None
    )
    result["pae_matrix"] = pae_matrix  # full matrix for downstream analysis
    return result


def run_af2_pae_filter(
    candidate_fasta: str,
    target_fasta: str,
    *,
    iptm_threshold: float = 0.6,
    pae_threshold: float = 15.0,
    af2ig_output_dir: str | None = None,
) -> FilterResult:
    """Evaluate AlphaFold2 Multimer PAE interaction score and iPTM.

    When ``af2ig_output_dir`` is provided (e.g. the output from a prior
    ``run_af2ig_filter`` call), this function re-uses those results without
    running ColabFold again.  Otherwise it runs a fresh ColabFold multimer
    prediction.

    Pass criteria (both must hold):
    - ``iptm ≥ iptm_threshold``   (default 0.6)
    - ``mean_inter_pae ≤ pae_threshold`` (default 15.0 Å)

    Args:
        candidate_fasta: Candidate binder in FASTA format.
        target_fasta: Target protein in FASTA format.
        iptm_threshold: Minimum iPTM to pass (default 0.6).
        pae_threshold: Maximum mean inter-chain PAE to pass (default 15.0 Å).
        af2ig_output_dir: Optional path to an existing AF2-IG output directory.
            When supplied, ColabFold is **not** re-run.

    Returns:
        FilterResult with ``filter_name="AF2-PAE"``.
    """
    candidate_seq = _extract_sequence(candidate_fasta)
    target_seq = _extract_sequence(target_fasta)

    if af2ig_output_dir:
        output_dir = Path(af2ig_output_dir)
        print(f"[AF2-PAE] Re-using AF2-IG output: {output_dir}")
    else:
        # Run a fresh ColabFold prediction
        output_dir = _make_output_dir("af2_pae")
        combined_seq = f"{candidate_seq}:{target_seq}"
        input_dir = output_dir / "tmp_input"
        input_dir.mkdir(parents=True, exist_ok=True)
        input_fasta = input_dir / "candidate_target_complex.fasta"
        input_fasta.write_text(
            f">candidate_target_complex\n{combined_seq}\n",
            encoding="utf-8",
        )

        print("[AF2-PAE] Running ColabFold multimer prediction (PAE mode)…")
        print(f"[AF2-PAE] Candidate length: {len(candidate_seq)}")
        print(f"[AF2-PAE] Target length: {len(target_seq)}")

        try:
            from colabfold.batch import get_queries, run
            from colabfold.utils import setup_logging
        except Exception as exc:
            return FilterResult(
                filter_name="AF2-PAE",
                passed=False,
                iptm=0.0,
                threshold=iptm_threshold,
                output_dir=str(output_dir),
                error=(
                    "ColabFold import failed. "
                    f"Install ColabFold in this environment. Details: {exc}"
                ),
            )

        try:
            setup_logging(output_dir / "log.txt")
            queries, is_complex = get_queries(str(input_dir))
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
                num_relax=0,
            )
        except Exception as exc:
            return FilterResult(
                filter_name="AF2-PAE",
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

    # Parse scores and PAE
    scores = _parse_af2_pae(output_dir, len(candidate_seq))

    iptm = scores.get("iptm", 0.0)
    mean_inter_pae = scores.get("mean_inter_pae")
    pae_note = scores.get("pae_note", "")

    # Pass requires both iPTM and PAE to meet thresholds.
    # If PAE is unavailable, fall back to iPTM-only pass.
    if mean_inter_pae is not None:
        passed = iptm >= iptm_threshold and mean_inter_pae <= pae_threshold
    else:
        passed = iptm >= iptm_threshold
        pae_note = pae_note or "PAE matrix unavailable; pass based on iPTM only"

    print(f"[AF2-PAE] iPTM: {iptm:.3f} (threshold: {iptm_threshold})")
    if mean_inter_pae is not None:
        print(f"[AF2-PAE] Mean inter-chain PAE: {mean_inter_pae:.2f} Å (threshold: ≤{pae_threshold} Å)")
    else:
        print(f"[AF2-PAE] Mean inter-chain PAE: N/A – {pae_note}")
    print(f"[AF2-PAE] Result: {'PASS ✅' if passed else 'FAIL ❌'}")

    raw: dict[str, Any] = {
        "iptm": iptm,
        "mean_inter_pae": mean_inter_pae,
        "pae_threshold": pae_threshold,
        "source_file": scores.get("source_file", ""),
    }
    if pae_note:
        raw["pae_note"] = pae_note
    if "ptm" in scores:
        raw["ptm"] = scores["ptm"]
    # Omit the full pae_matrix from raw_scores to keep the trace compact;
    # it is available via the source_file.

    return FilterResult(
        filter_name="AF2-PAE",
        passed=passed,
        iptm=iptm,
        threshold=iptm_threshold,
        output_dir=str(output_dir),
        raw_scores=raw,
    )


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
    # ── existing filters ──────────────────────────────────────────────────
    enable_af2ig: bool = True,
    enable_protenix: bool = True,
    af2ig_threshold: float = 0.6,
    protenix_threshold: float = 0.7,
    protenix_seeds: int = 3,
    # ── chemical filter ───────────────────────────────────────────────────
    enable_chemical: bool = True,
    chemical_skip_lipinski_for_peptides: bool = True,
    # ── ESM2 expression metric ────────────────────────────────────────────
    enable_esm2: bool = True,
    esm2_model: str = "facebook/esm2_t6_8M_UR50D",
    esm2_ll_threshold: float = -1.5,
    # ── Rosetta ΔSASA docking ─────────────────────────────────────────────
    enable_rosetta_sasa: bool = False,  # off by default – requires PyRosetta
    rosetta_sasa_threshold: float = 500.0,
    # ── AF2 PAE interaction + iPTM ────────────────────────────────────────
    enable_af2_pae: bool = True,
    af2_pae_iptm_threshold: float = 0.6,
    af2_pae_threshold: float = 15.0,
) -> list[FilterResult]:
    """Run all enabled structural and biochemical validation filters.

    Returns a list of FilterResult objects. The candidate is considered
    rejected if ANY enabled filter fails (strict mode).

    Filter execution order:
    1. Chemical (RDKit Lipinski Ro5 + PAINS) — fast, no GPU required.
    2. ESM2 log-likelihood expression metric — fast, CPU-friendly.
    3. AF2-IG (ColabFold multimer, iPTM) — slow, requires ColabFold.
    4. AF2-PAE (inter-chain PAE, reuses AF2-IG output if available).
    5. Protenix (iPTM) — slow, requires protenix CLI.
    6. Rosetta ΔSASA — slow, requires PyRosetta; disabled by default.

    Args:
        candidate_fasta: Candidate binder in FASTA format.
        target_fasta: Target protein in FASTA format.
        enable_af2ig: Whether to run the AF2-IG iPTM filter.
        enable_protenix: Whether to run the Protenix filter.
        af2ig_threshold: iPTM threshold for AF2-IG.
        protenix_threshold: iPTM threshold for Protenix.
        protenix_seeds: Number of Protenix prediction seeds.
        enable_chemical: Whether to run the RDKit chemical filter.
        chemical_skip_lipinski_for_peptides: Skip Lipinski for peptide-like
            molecules when True.
        enable_esm2: Whether to run the ESM2 log-likelihood filter.
        esm2_model: ESM2 model checkpoint name.
        esm2_ll_threshold: Minimum mean log-likelihood to pass.
        enable_rosetta_sasa: Whether to run the Rosetta ΔSASA filter.
            Disabled by default – requires PyRosetta.
        rosetta_sasa_threshold: Minimum ΔSASA (Å²) to pass.
        enable_af2_pae: Whether to run the AF2 PAE interaction filter.
            When AF2-IG is also enabled, reuses its output directory.
        af2_pae_iptm_threshold: Minimum iPTM for the AF2-PAE filter.
        af2_pae_threshold: Maximum mean inter-chain PAE (Å) to pass.
    """
    results: list[FilterResult] = []

    # ── 1. Chemical filter ────────────────────────────────────────────────
    if enable_chemical:
        print("\n" + "=" * 60)
        print("STRUCTURAL FILTER: Chemical (RDKit Lipinski Ro5 + PAINS)")
        print("=" * 60)
        try:
            chem_result = run_chemical_filter(
                candidate_fasta,
                skip_lipinski_for_peptides=chemical_skip_lipinski_for_peptides,
            )
            results.append(chem_result)
        except Exception as exc:
            print(f"[Chemical-RDKit] Unexpected error: {exc}")
            results.append(FilterResult(
                filter_name="Chemical-RDKit",
                passed=False,
                iptm=0.0,
                threshold=0.5,
                error=str(exc),
            ))

    # ── 2. ESM2 log-likelihood expression metric ──────────────────────────
    if enable_esm2:
        print("\n" + "=" * 60)
        print("STRUCTURAL FILTER: ESM2 Log-Likelihood (Expression Metric)")
        print("=" * 60)
        try:
            esm2_result = run_esm2_ll_filter(
                candidate_fasta,
                model_name=esm2_model,
                ll_threshold=esm2_ll_threshold,
            )
            results.append(esm2_result)
        except Exception as exc:
            print(f"[ESM2-LogLikelihood] Unexpected error: {exc}")
            results.append(FilterResult(
                filter_name="ESM2-LogLikelihood",
                passed=False,
                iptm=0.0,
                threshold=esm2_ll_threshold,
                error=str(exc),
            ))

    # ── 3. AF2-IG filter ──────────────────────────────────────────────────
    af2ig_output_dir: str | None = None
    if enable_af2ig:
        print("\n" + "=" * 60)
        print("STRUCTURAL FILTER: AF2-IG (ColabFold)")
        print("=" * 60)
        try:
            af2ig_result = run_af2ig_filter(
                candidate_fasta, target_fasta, af2ig_threshold
            )
            results.append(af2ig_result)
            # Capture output dir so AF2-PAE can reuse it
            af2ig_output_dir = af2ig_result.output_dir or None
        except FileNotFoundError as exc:
            print(f"[AF2-IG] Skipped: {exc}")
            results.append(FilterResult(
                filter_name="AF2-IG",
                passed=False,
                iptm=0.0,
                threshold=af2ig_threshold,
                error=str(exc),
            ))

    # ── 4. AF2 PAE interaction + iPTM ─────────────────────────────────────
    if enable_af2_pae:
        print("\n" + "=" * 60)
        print("STRUCTURAL FILTER: AF2 Multimer PAE Interaction + iPTM")
        print("=" * 60)
        try:
            af2_pae_result = run_af2_pae_filter(
                candidate_fasta,
                target_fasta,
                iptm_threshold=af2_pae_iptm_threshold,
                pae_threshold=af2_pae_threshold,
                af2ig_output_dir=af2ig_output_dir,  # reuse if available
            )
            results.append(af2_pae_result)
        except Exception as exc:
            print(f"[AF2-PAE] Unexpected error: {exc}")
            results.append(FilterResult(
                filter_name="AF2-PAE",
                passed=False,
                iptm=0.0,
                threshold=af2_pae_iptm_threshold,
                error=str(exc),
            ))

    # ── 5. Protenix filter ────────────────────────────────────────────────
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

    # ── 6. Rosetta ΔSASA docking ──────────────────────────────────────────
    if enable_rosetta_sasa:
        print("\n" + "=" * 60)
        print("STRUCTURAL FILTER: Rosetta ΔSASA Docking")
        print("=" * 60)
        try:
            rosetta_result = run_rosetta_sasa_filter(
                candidate_fasta,
                target_fasta,
                sasa_threshold=rosetta_sasa_threshold,
            )
            results.append(rosetta_result)
        except Exception as exc:
            print(f"[Rosetta-SASA] Unexpected error: {exc}")
            results.append(FilterResult(
                filter_name="Rosetta-SASA",
                passed=False,
                iptm=0.0,
                threshold=rosetta_sasa_threshold,
                error=str(exc),
            ))

    return results
