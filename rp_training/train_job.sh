#!/bin/bash
#SBATCH --job-name=synthetic-train
#SBATCH --partition=bigbatch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=0                    # grab all node memory; adjust if needed
#SBATCH --time=04:00:00
#SBATCH --output=/home-mscluster/%u/slurm.%x.%N.%j.out
#SBATCH --error=/home-mscluster/%u/slurm.%x.%N.%j.err
# #SBATCH --gres=gpu:1             # uncomment if you want a GPU

set -euo pipefail

# ---- Paths (use absolute or --chdir below) ----
# If your notebook lives elsewhere, either:
#   a) set --chdir, or b) set NOTEBOOK to an absolute path
# Example with --chdir (recommended):
# SBATCH --chdir=/home-mscluster/$USER/rp_training

NOTEBOOK="training_implementation_new.ipynb"
RUN_DIR="$PWD"
OUT_DIR="$RUN_DIR/outputs"
OUT_NOTEBOOK="$OUT_DIR/synthetic_run_${SLURM_JOB_ID}.ipynb"

mkdir -p "$OUT_DIR"

echo "[info] Job: $SLURM_JOB_ID on $(hostname)"
echo "[info] Working dir: $RUN_DIR"
echo "[info] Notebook in:  $NOTEBOOK"
echo "[info] Notebook out: $OUT_NOTEBOOK"

# ---- Activate conda env ----
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate rp_env

# Prefer env libs over system ones (no fancy quoting)
if [ -z "${CONDA_PREFIX:-}" ]; then
  echo "[error] CONDA_PREFIX is empty; is the env activated?" >&2
  exit 2
fi

# 1) Ensure env has a modern libittnotify
conda install -y -c conda-forge ittapi

export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export LD_PRELOAD="$CONDA_PREFIX/lib/libittnotify.so:${LD_PRELOAD:-}"

# If your site autoloads Intel tools, this helps:
# module purge 2>/dev/null || true
# unset VTUNE_HOME AMPLXE_DISABLE CPATH LIBRARY_PATH  # avoid old Intel paths


# ---- Ensure tools & kernel exist ----
python - <<'PY'
import sys, subprocess
def ensure(pkgs):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *pkgs])
try:
    import nbconvert, ipykernel  # noqa
except Exception:
    ensure(["nbconvert", "ipykernel"])
# Register a kernelspec for this env (idempotent)
subprocess.call([sys.executable, "-m", "ipykernel", "install", "--user",
                 "--name", "rp_env", "--display-name", "Python (rp_env)"])
PY

# reduces fragmentation
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ---- Execute notebook with nbconvert ----
jupyter nbconvert \
  --to notebook \
  --execute "$NOTEBOOK" \
  --output "$(basename "$OUT_NOTEBOOK")" \
  --output-dir "$OUT_DIR" \
  --ExecutePreprocessor.kernel_name="rp_env" \
  --ExecutePreprocessor.timeout=-1

# ---- Verify it exists and has outputs ----
ls -lah "$OUT_NOTEBOOK"
python - <<PY
import json, sys
p = r"$OUT_NOTEBOOK"
nb = json.load(open(p))
cells = nb.get("cells", [])
n_with = sum(1 for c in cells if c.get("outputs"))
n_out  = sum(len(c.get("outputs", [])) for c in cells)
print(f"[verify] cells with outputs: {n_with}, total outputs: {n_out}")
PY

echo "[done] Executed notebook saved to: $OUT_NOTEBOOK"

