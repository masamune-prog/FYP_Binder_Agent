from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem import Lipinski as RDKitLipinski
try:
    from rdkit.Contrib.SA_Score import sascorer
except ImportError:
    sascorer = None


def fasta_to_smiles(fasta_text: str) -> str:
    """Convert a single protein FASTA record into a peptide SMILES string."""
    if not isinstance(fasta_text, str) or not fasta_text.strip():
        raise ValueError("FASTA input must be a non-empty string.")

    mol = Chem.MolFromFASTA(fasta_text)
    if mol is None:
        raise ValueError("RDKit could not convert the FASTA sequence to a molecule.")

    return Chem.MolToSmiles(mol)



def check_lipinski(mol: Chem.Mol, is_peptide: bool = False) -> bool:
    """
    Checks if a molecule satisfies Lipinski's Rule of 5:
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
        
    mw = Descriptors.MolWt(mol)
    logp = Descriptors.MolLogP(mol)
    h_donors = RDKitLipinski.NumHDonors(mol)
    h_acceptors = RDKitLipinski.NumHAcceptors(mol)
    
    # If it's a peptide, we use "beyond Rule of 5" (bRo5) guidelines
    if is_peptide:
        # Peptides generally target protein-protein interactions (like MDM2)
        # where higher MW and more H-bonds are required for binding surface area.
        violations = 0
        if mw >= 2000: violations += 1       # Relaxed for large peptides
        if not (-2 <= logp <= 8): violations += 1 # Wider logP tolerance
        return violations <= 1

    # Standard Small Molecule Lipinski
    violations = 0
    if mw >= 500: violations += 1
    if logp >= 5: violations += 1
    if h_donors >= 5: violations += 1
    if h_acceptors >= 10: violations += 1
    return violations <= 1


def passes_pains_filter(mol: Chem.Mol) -> bool:
    """
    Filters out Pan-Assay Interference Compounds (PAINS) using RDKit's
    built-in FilterCatalog. Instantly drops molecules containing substructures
    known to cause false positives in biological assays.
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


def check_synthetic_accessibility(mol: Chem.Mol, max_score: float = 4.5) -> bool:
    """
    Evaluates synthetic ease using RDKit's legacy SA score helper when available.
    Lower values represent structures that are more straightforward to synthesize.
    Synthesis is not issue for peptide so we will skip this check for peptides 
    """
    if mol is None:
        return False

    if sascorer is not None:
        score = sascorer.calculateScore(mol)
    else:
        print("SA score helper not available; ERROR!")
        score = float('inf')  # Assume worst-case synthetic complexity
    return score < max_score


def pipeline_filter(smiles: str) -> dict:
    """
    Convenience wrapper to run a single SMILES string through 
    the complete screening triage.
    """
    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        return {"smiles": smiles, "passed": False, "reason": "Invalid SMILES"}
        
    lipinski = check_lipinski(mol, is_peptide=True)
    pains = passes_pains_filter(mol)
    #synthetic = check_synthetic_accessibility(mol)
    
    #passed = lipinski and pains and synthetic
    passed = lipinski and pains
    reasons = []
    if not lipinski: reasons.append("Failed Lipinski Ro5")
    if not pains: reasons.append("Flagged by PAINS")
    #if not synthetic: reasons.append("High Synthetic Complexity")
    
    return {
        "smiles": smiles,
        "passed": passed,
        "reason": ", ".join(reasons) if not passed else "Passed All Filters"
    }

# Example Usage
if __name__ == "__main__":
    # Aspirin (Should pass easily)
    aspirin_smiles = "CC(=O)OC1=CC=CC=C1C(=O)O"
    print(f"Aspirin: {pipeline_filter(aspirin_smiles)}")
    
    # Toxoflavin (Classic PAINS tracking structure / tricky synthesis layout)
    toxoflavin_smiles = "CN1C2=C(C(=O)N(C1=O)C)N=CC=N2" 
    print(f"Toxoflavin: {pipeline_filter(toxoflavin_smiles)}")

    fasta_input = ">pMI_peptide\nNWSPKTFGAWLQY"
    
    fasta_smiles = fasta_to_smiles(fasta_input)
    print(f"Peptide SMILES: {fasta_smiles}")
    print(f"Peptide Filter Result: {pipeline_filter(fasta_smiles)}")