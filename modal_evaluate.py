"""
Modal Deployment for GNN Subgraph Evaluation

This script runs the full evaluation pipeline on Modal's cloud infrastructure.
It handles:
- Uploading models from local `models/` directory
- Mounting corrected `src/` code into the container
- Running evaluate.py with proper arguments for each dataset
- Auto-downloading result CSVs locally

Usage:
    # Run ALL datasets (cora, arxiv, mag) end-to-end:
    modal run modal_evaluate.py

    # Run a single dataset:
    modal run modal_evaluate.py --dataset arxiv

    # Upload models only (first-time setup):
    modal run modal_evaluate.py --setup-only

    # Download results only:
    modal run modal_evaluate.py --download
"""

import modal
import os

# Define the Modal app
app = modal.App("gnn-subgraph-eval")

# Create a volume for persisting data, models, and results
volume = modal.Volume.from_name("gnn-data-volume", create_if_missing=True)

# Define the container image with all dependencies
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "cmake", "build-essential")  # For SubgraphMatching build
    .pip_install(
        "torch==2.1.2",
        "numpy<2.0.0",  # Pin numpy for compatibility
    )
    .pip_install(
        "torch-geometric",
        "torch-sparse",
        "torch-scatter",
        "torch-cluster",
        "ogb",
        "pymetis==2022.1",  # Match training script version to avoid segfaults
        "scipy",
        "networkx",
        "matplotlib",
        "faiss-cpu>=1.7.4",
        "pandas>=2.0.0",
        "tqdm>=4.65.0",
        "vf3py",  # VF3 subgraph isomorphism solver
    )
    # Build SubgraphMatching binary (DP-iso, CFL, TurboISO)
    .run_commands(
        "git clone https://github.com/RapidsAtHKUST/SubgraphMatching.git /app/SubgraphMatching",
        # Patch config.h: SI 2 = scalar (no AVX2), HYBRID 1 = merge-based (no AVX2 galloping), enable failing set
        "sed -i 's/#define SI 0/#define SI 2/' /app/SubgraphMatching/configuration/config.h",
        "sed -i 's/#define HYBRID 0/#define HYBRID 1/' /app/SubgraphMatching/configuration/config.h",
        "sed -i 's|// #define ENABLE_FAILING_SET|#define ENABLE_FAILING_SET|' /app/SubgraphMatching/configuration/config.h",
        # Remove -march=native, use x86-64-v2 for portability (SSE4.2 but no AVX2)
        "sed -i 's/-march=native/-march=x86-64-v2/' /app/SubgraphMatching/CMakeLists.txt",
        "cd /app/SubgraphMatching && mkdir -p build && cd build && cmake .. && make -j$(nproc)",
    )
)

# Volume mount path inside the container
VOLUME_PATH = "/data"

# All datasets to benchmark
ALL_DATASETS = ["cora", "arxiv", "mag"]


# ---------------------------------------------------------------------------
# 1) Upload: Mount local models/ dir and copy .pth files into the volume
# ---------------------------------------------------------------------------
@app.function(
    image=image.add_local_dir("models", remote_path="/root/local_models"),
    volumes={VOLUME_PATH: volume},
    timeout=3600,
)
def upload_models():
    """Upload local model checkpoints into the persistent Modal volume."""
    import shutil

    for subdir in ("models", "cache", "results"):
        os.makedirs(f"{VOLUME_PATH}/{subdir}", exist_ok=True)

    local_dir = "/root/local_models"
    vol_dir = f"{VOLUME_PATH}/models"

    if not os.path.exists(local_dir):
        print("⚠ No local models/ directory found!")
        volume.commit()
        return "No models to upload"

    uploaded = 0
    for fname in sorted(os.listdir(local_dir)):
        if not fname.endswith(".pth"):
            continue
        src = os.path.join(local_dir, fname)
        dst = os.path.join(vol_dir, fname)
        src_size = os.path.getsize(src)

        if os.path.exists(dst) and os.path.getsize(dst) == src_size:
            print(f"  ✓ {fname} already on volume ({src_size:,} bytes)")
            continue

        print(f"  📦 Uploading {fname} ({src_size:,} bytes) ...")
        shutil.copy2(src, dst)
        uploaded += 1

    volume.commit()
    msg = f"✅ Upload complete – {uploaded} new file(s) copied"
    print(msg)
    return msg


# ---------------------------------------------------------------------------
# 2) Evaluate: Mount local src/ so the corrected code actually runs
# ---------------------------------------------------------------------------
@app.function(
    image=image.add_local_dir("src", remote_path="/app/src"),
    volumes={VOLUME_PATH: volume},
    timeout=3600 * 10,  # 10-hour timeout for large datasets
    cpu=8,
    memory=32768,  # 32 GB RAM
    gpu="T4",
)
def run_evaluation(
    dataset: str = "cora",
    target_queries: int = 10,
    top_k: int = 20,
    skip_solver: bool = False,
    run_baseline: bool = False,
    query_types: str = None,
    solver: str = 'vf3',
    run_full_graph: bool = False,
):
    """
    Run src.evaluate on Modal with the corrected evaluation code.

    Models are read from the persistent volume (/data/models/).
    Results are written to the persistent volume (/data/results/).
    """
    import subprocess
    import sys

    os.chdir("/app")  # src/ is mounted here as /app/src/

    # Build the command -------------------------------------------------------
    output_csv = f"{VOLUME_PATH}/results/{dataset}_eval.csv"
    cmd = [
        sys.executable, "-u", "-m", "src.evaluate",
        "--dataset", dataset,
        "--target_queries", str(target_queries),
        "--top_k", str(top_k),
        "--output", output_csv,
    ]

    model_path = f"{VOLUME_PATH}/models/{dataset}-6_layer-model-jigsaw.pth"
    hierarchy_path = f"{VOLUME_PATH}/cache/{dataset}_hierarchy.pkl"

    if os.path.exists(model_path):
        cmd.extend(["--model_path", model_path])
        print(f"✓ Model found: {model_path}")
    else:
        print(f"⚠ Model NOT found at {model_path} — using default")

    # Always pass hierarchy_cache so it gets saved after building
    os.makedirs(os.path.dirname(hierarchy_path), exist_ok=True)
    cmd.extend(["--hierarchy_cache", hierarchy_path])
    if os.path.exists(hierarchy_path):
        print(f"✓ Hierarchy cache found: {hierarchy_path}")
    else:
        print(f"ℹ No hierarchy cache — will build and save to {hierarchy_path}")

    if skip_solver:
        cmd.append("--skip_solver")
    if run_baseline:
        cmd.append("--run_baseline")
    if query_types:
        cmd.extend(["--query_types", query_types])
    if solver:
        cmd.extend(["--solver", solver])
    if run_full_graph:
        cmd.append("--run_full_graph")

    print(f"\n🚀 Running: {' '.join(cmd)}\n")

    # Run ---------------------------------------------------------------------
    result = subprocess.run(cmd, capture_output=False)

    # Persist results to volume
    volume.commit()

    return result.returncode


# ---------------------------------------------------------------------------
# 3) Download: pull a result CSV from the volume and return it as text
# ---------------------------------------------------------------------------
@app.function(
    image=image,
    volumes={VOLUME_PATH: volume},
)
def download_result(dataset: str = "cora", filename: str = None):
    """Read a result file from the volume and return its contents."""
    if filename is None:
        filename = f"{dataset}_eval.csv"
    path = f"{VOLUME_PATH}/results/{filename}"
    if os.path.exists(path):
        with open(path, "r") as f:
            return f.read()
    return None


# ---------------------------------------------------------------------------
# Local entrypoint – orchestrates everything from your laptop
# ---------------------------------------------------------------------------
@app.local_entrypoint()
def main(
    dataset: str = "all",
    target_queries: int = 10,
    top_k: int = 20,
    skip_solver: bool = False,
    run_baseline: bool = False,
    setup_only: bool = False,
    download: bool = False,
    query_types: str = None,
    solver: str = 'vf3',
    run_full_graph: bool = False,
):
    """
    End-to-end Modal CLI entrypoint.

    Examples:
        # Run ALL datasets (upload models → evaluate → download CSVs):
        modal run modal_evaluate.py

        # Run a single dataset:
        modal run modal_evaluate.py --dataset arxiv

        # Upload models only:
        modal run modal_evaluate.py --setup-only

        # Download previously-generated results:
        modal run modal_evaluate.py --download
    """
    datasets = ALL_DATASETS if dataset == "all" else [dataset]

    # ------------------------------------------------------------------
    # Setup-only mode: just upload models
    # ------------------------------------------------------------------
    if setup_only:
        print("📦 Uploading models to Modal volume ...")
        result = upload_models.remote()
        print(result)
        return

    # ------------------------------------------------------------------
    # Download-only mode: pull CSVs from volume
    # ------------------------------------------------------------------
    if download:
        for ds in datasets:
            csv_text = download_result.remote(ds)
            if csv_text:
                out = f"{ds}_eval.csv"
                with open(out, "w") as f:
                    f.write(csv_text)
                print(f"✓ Downloaded {out}")
            else:
                print(f"⚠ No results found for {ds}")
        return

    # ------------------------------------------------------------------
    # Full pipeline: upload models → run benchmarks → download CSVs
    # ------------------------------------------------------------------
    print("=" * 60)
    print("  GNN Subgraph Evaluation — Modal Pipeline")
    print("=" * 60)
    print(f"  Datasets     : {', '.join(datasets)}")
    print(f"  Queries/type : {target_queries}")
    print(f"  Top-k        : {top_k}")
    print(f"  Skip Solver  : {skip_solver}")
    print(f"  Solver       : {solver}")
    print(f"  Baseline     : {run_baseline}")
    print(f"  Full Graph   : {run_full_graph}")
    print(f"  Query Types  : {query_types or 'all'}")
    print("=" * 60)

    # Step 1: Upload models
    print("\n📦 Step 1/3: Uploading models ...")
    upload_msg = upload_models.remote()
    print(f"   {upload_msg}")

    # Step 2: Run evaluations
    print(f"\n🚀 Step 2/3: Running evaluations ...")
    results = {}
    for ds in datasets:
        print(f"\n{'─'*40}")
        print(f"   Evaluating: {ds}")
        print(f"{'─'*40}")
        exit_code = run_evaluation.remote(
            dataset=ds,
            target_queries=target_queries,
            top_k=top_k,
            skip_solver=skip_solver,
            run_baseline=run_baseline,
            query_types=query_types,
            solver=solver,
            run_full_graph=run_full_graph,
        )
        results[ds] = exit_code
        if exit_code == 0:
            print(f"   ✅ {ds} completed successfully")
        else:
            print(f"   ❌ {ds} failed (exit code {exit_code})")

    # Step 3: Download results
    print(f"\n📥 Step 3/3: Downloading result CSVs ...")
    for ds in datasets:
        if results[ds] != 0:
            print(f"   ⏭ Skipping {ds} (evaluation failed)")
            continue
        csv_text = download_result.remote(ds)
        if csv_text:
            out = f"{ds}_eval.csv"
            with open(out, "w") as f:
                f.write(csv_text)
            print(f"   ✓ Saved {out}")
        else:
            print(f"   ⚠ No results file for {ds}")
        
        # Also download summary text file
        summary_text = download_result.remote(ds, filename=f"{ds}_eval_summary.txt")
        if summary_text:
            out_txt = f"{ds}_eval_summary.txt"
            with open(out_txt, "w") as f:
                f.write(summary_text)
            print(f"   ✓ Saved {out_txt}")

    # Summary
    print("\n" + "=" * 60)
    print("  Summary")
    print("=" * 60)
    for ds, code in results.items():
        status = "✅ PASS" if code == 0 else "❌ FAIL"
        print(f"   {ds:>8s}  {status}")
    print("=" * 60)
