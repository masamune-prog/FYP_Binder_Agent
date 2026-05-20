import os
from pathlib import Path
from colabfold.batch import get_queries, run
from colabfold.utils import setup_logging

def run_binding_screening(candidate_seq, target_seq, job_id, out_dir="./results"):
    # 1. Setup workspace directories
    input_dir = Path("./tmp_input")
    output_dir = Path(out_dir) / job_id
    input_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 2. Programmatically construct the multi-chain FASTA format
    fasta_path = input_dir / f"{job_id}.fasta"
    with open(fasta_path, "w") as f:
        f.write(f">{job_id}\n{candidate_seq}:{target_seq}\n")
    
    # 3. Setup Colabfold logger
    setup_logging(output_dir / "log.txt")
    
    # 4. Parse queries out of the temporary directory
    queries, is_complex = get_queries(str(input_dir))
    
    # 5. Run the AlphaFold-Multimer Inference Engine
    run(
        queries=queries,
        result_dir=str(output_dir),
        is_complex=is_complex,
        use_bfloat16=False,
        use_templates=False,               # Toggle True if you want structural template lookups
        msa_mode="MMseqs2 (UniRef+Environmental)",
        model_type="alphafold2_multimer_v3",
        num_models=5,                      # Predicts all 5 standard AF-Multimer models
        num_recycles=6,                    # Structural recycles per model
        num_relax=1,                       # Energy-minimizes the single best-ranked structural model
        relax_max_iterations=2000
    )
    
    # Clean up input file for clean iterative batching
    os.remove(fasta_path)
    print(f"Prediction for {job_id} complete. Files saved to {output_dir}")

# --- Example Execution Execution ---
if __name__ == "__main__":
    candidate_seq = "WGRFSTLKMNAYPQIDLTFGHVRCMEKTLPNSWQHIYDFGSP"
    target_seq = "MFVFLVLLPLVSSQCVNLTTRTQLPPAYTNSFTRGVYYPDKVFRSSVLHSTQDLFLPFFSNVTWFHAIHVSGTNGTKRFDNPVLPFNDGVYFASTEKSNIIRGWIFGTTLDSKTQSLLIVNNATNVVIKVCEFQFCNDPFLGVYYHKNNKSWMESEFRVYSSANNCTFEYVSQPFLMDLEGKQGNFKNLREFVFKNIDGYFKIYSKHTPINLVRDLPQGFSALEPLVDLPIGINITRFQTLLALHRSYLTPGDSSSGWTAGAAAYYVGYLQPRTFLLKYNENGTITDAVDCALDPLSETKCTLKSFTVEKGIYQTSNFRVQPTESIVRFPNITNLCPFGEVFNATRFASVYAWNRKRISNCVADYSVLYNSASFSTFKCYGVSPTKLNDLCFTNVYADSFVIRGDEVRQIAPGQTGKIADYNYKLPDDFTGCVIAWNSNNLDSKVGGNYNYLYRLFRKSNLKPFERDISTEIYQAGSTPCNGVEGFNCYFPLQSYGFQPTNGVGYQPYRVVVLSFELLHAPATVCGPKKSTNLVKNKCVNFNFNGLTGTGVLTESNKKFLPFQQFGRDIADTTDAVRDPQTLEILDITPCSFGGVSVITPGTNTSNQVAVLYQDVNCTEVPVAIHADQLTPTWRVYSTGSNVFQTRAGCLIGAEHVNNSYECDIPIGAGICASYQTQTNSPRRARSVASQSIIAYTMSLGAENSVAYSNNSIAIPTNFTISVTTEILPVSMTKTSVDCTMYICGDSTECSNLLLQYGSFCTQLNRALTGIAVEQDKNTQEVFAQVKQIYKTPPIKDFGGFNFSQILPDPSKPSKRSFIEDLLFNKVTLADAGFIKQYGDCLGDIAARDLICAQKFNGLTVLPPLLTDEMIAQYTSALLAGTITSGWTFGAGAALQIPFAMQMAYRFNGIGVTQNVLYENQKLIANQFNSAIGKIQDSLSSTASALGKLQDVVNQNAQALNTLVKQLSSNFGAISSVLNDILSRLDKVEAEVQIDRLITGRLQSLQTYVTQQLIRAAEIRASANLAATKMSECVLGQSKRVDFCGKGYHLMSFPQSAPHGVVFLHVTYVPAQEKNFTTAPAICHDGKAHFPREGVFVSNGTHWFVTQRNFYEPQIITTDNTFVSGNCDVVIGIVNNTVYDPLQPELDSFKEELDKYFKNHTSPDVDLGDISGINASVVNIQKEIDRLNEVAKNLNESLIDLQELGKYEQYIKWPWYIWLGFIAGLIAIVMVTIMLCCMTSCCSCLKGCCSCGSCCKFDEDDSEPVLKGVKLHYT"

    
    run_binding_screening(
        candidate_seq=candidate_seq,
        target_seq=target_seq,
        job_id="candidate_run_01"
    )