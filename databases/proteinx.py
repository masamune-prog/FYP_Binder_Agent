import os
import json
import glob
import subprocess
import pandas as pd
import numpy as np
import modal

# 1. Image definition with system compilers and curl/wget for downloads
image = (
    modal.Image.from_registry("nvidia/cuda:12.1.1-devel-ubuntu22.04", add_python="3.12")
    .apt_install("g++", "ninja-build", "clang", "wget")
    .env({
        "CUDA_HOME": "/usr/local/cuda",
        "PATH": "/usr/local/cuda/bin:$PATH",
        "LD_LIBRARY_PATH": "/usr/local/cuda/lib64:$LD_LIBRARY_PATH",
        "CC": "clang",
        # CRITICAL: Point Protenix directly to our persistent storage volume
        "PROTENIX_ROOT_DIR": "/data/protenix_assets"
    })
    .pip_install("pandas", "numpy", "openpyxl", "requests", "rdkit", "scikit-learn")
    .pip_install("scikit-learn-extra") 
    .pip_install("torch", pre=True)
    .pip_install("protenix") 
)

volume = modal.Volume.from_name("protenix-storage", create_if_missing=True)
app = modal.App("protenix-batch-inference", image=image)


# 2. Asset Cache Sync (Runs once on CPU to prime the storage volume)
@app.function(volumes={"/data": volume}, timeout=1200)
def download_assets_once():
    """Ensures all structural databases and model checkpoints exist in the Volume."""
    base_dir = "/data/protenix_assets"
    os.makedirs(f"{base_dir}/common", exist_ok=True)
    os.makedirs(f"{base_dir}/checkpoint", exist_ok=True)
    
    urls = {
        f"{base_dir}/common/components.cif": "https://protenix.tos-cn-beijing.volces.com/common/components.cif",
        f"{base_dir}/common/components.cif.rdkit_mol.pkl": "https://protenix.tos-cn-beijing.volces.com/common/components.cif.rdkit_mol.pkl",
        f"{base_dir}/common/clusters-by-entity-40.txt": "https://protenix.tos-cn-beijing.volces.com/common/clusters-by-entity-40.txt",
        f"{base_dir}/common/obsolete_release_date.csv": "https://protenix.tos-cn-beijing.volces.com/common/obsolete_release_date.csv",
        f"{base_dir}/checkpoint/protenix_base_20250630_v1.0.0.pt": "https://protenix.tos-cn-beijing.volces.com/checkpoint/protenix_base_20250630_v1.0.0.pt"
    }
    
    for local_path, url in urls.items():
        if not os.path.exists(local_path) or os.path.getsize(local_path) == 0:
            print(f"Downloading {os.path.basename(local_path)} to Modal Volume...")
            # Use wget with robust timeout settings to handle drops
            subprocess.run(["wget", "--timeout=30", "--tries=5", "-O", local_path, url], check=True)
        else:
            print(f"✅ {os.path.basename(local_path)} already cached.")
    
    volume.commit()
    print("✨ Protenix data assets are verified and ready inside your Volume!")


# 3. Remote GPU Inference Worker
@app.function(gpu="L4", volumes={"/data": volume}, timeout=3600)
def run_protenix_row(index: int, target_sequence: str, binder_sequence: str):
    INPUT_JSON_PATH = f"/tmp/input_{index}.json"
    OUTPUT_DIR = f"/data/output_batch/sample_row_{index}"
    BINDER_CHAIN = "B"
    TARGET_CHAIN = "A"

    input_data = [
        {
            "id": f"sample_row_{index}",
            "name": f"sample_row_{index}",
            "sequences": [
                {"proteinChain": {"sequence": target_sequence, "count": 1}},
                {"proteinChain": {"sequence": binder_sequence, "count": 1}}
            ]
        }
    ]
    
    os.makedirs(os.path.dirname(INPUT_JSON_PATH), exist_ok=True)
    with open(INPUT_JSON_PATH, 'w') as f:
        json.dump(input_data, f, indent=4)

    cmd = [
        "protenix", "pred",
        "-i", INPUT_JSON_PATH,
        "-o", OUTPUT_DIR,
        "--model_name", "protenix_base_20250630_v1.0.0",
        "--dtype", "bf16",
        "--use_default_params", "true", 
        "--trimul_kernel", "torch",
        "--triatt_kernel", "torch",
        "--enable_cache", "true",
        "--enable_fusion", "true",
    ]

    print(f"Starting remote inference for Row {index}...")
    try:
        # Capture logs natively so we can view streaming progress outputs live
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Protenix run failed on Row {index}!")
        print(e.stderr)
        return np.nan, np.nan

    predictions_pattern = os.path.join(OUTPUT_DIR, f"sample_row_{index}", "seed_*", "predictions", "*_summary_confidence_sample_*.json")
    summary_files = glob.glob(predictions_pattern)
    
    row_ptms, row_iptms = [], []
    for file_path in summary_files:
        with open(file_path, "r") as f:
            try:
                data = json.load(f)
                row_ptms.append(data.get("chain_ptm", {}).get(BINDER_CHAIN, 0.0))
                binder_iptm = data.get("chain_pair_iptm", {}).get(TARGET_CHAIN, {}).get(BINDER_CHAIN, 0.0)
                if binder_iptm == 0.0:
                    binder_iptm = data.get("chain_pair_iptm", {}).get(BINDER_CHAIN, {}).get(TARGET_CHAIN, 0.0)
                row_iptms.append(binder_iptm)
            except Exception as parser_err:
                print(f"Parsing error: {parser_err}")

    if row_ptms and row_iptms:
        return float(np.mean(row_ptms)), float(np.mean(row_iptms))
    return np.nan, np.nan


# 4. Local Execution Runner
@app.local_entrypoint()
def main():
    # Verify/download files to volume storage backbone first
    print("Verifying structural cache assets in Modal volume...")
    download_assets_once.remote()

    csv_path = "plain_training.csv" 
    if not os.path.exists(csv_path):
        print(f"Please put your '{csv_path}' dataset file in this local folder first.")
        return

    df = pd.read_csv(csv_path)
    test_df = df.head(5).copy()
    
    avg_binder_ptms, avg_binder_iptms = [], []

    for index, row in test_df.iterrows():
        print(f"\n--- Processing Row {index+1}/{len(test_df)} ---")
        ptm, iptm = run_protenix_row.remote(
            index=index,
            target_sequence=row["target_sequence"],
            binder_sequence=row["sequence"]
        )
        print(f"Row {index} Output -> PTM: {ptm} | ipTM: {iptm}")
        avg_binder_ptms.append(ptm)
        avg_binder_iptms.append(iptm)

    test_df["avg_binder_ptm"] = avg_binder_ptms
    test_df["avg_binder_iptm"] = avg_binder_iptms
    test_df.to_csv("protenix_modal_evaluated.csv", index=False)
    print("\nCompleted! Locally saved out to 'protenix_modal_evaluated.csv'")