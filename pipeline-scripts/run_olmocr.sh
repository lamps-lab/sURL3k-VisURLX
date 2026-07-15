#!/bin/bash
#SBATCH --job-name=olmocr
#SBATCH -p high-gpu-mem
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=128G
#SBATCH --time=7-00:00:00
#SBATCH --output=logs/olmocr_%j.out
#SBATCH --error=logs/olmocr_%j.err

set -euo pipefail

# Edit these for your environment.
CONDA_ENV=olmocr
INPUT_DIR=./pdfs
WORKSPACE=./olmocr_workspace
CHUNK_SIZE=1
WORKERS=1

mkdir -p logs
source "$(conda info --base)/bin/activate" "$CONDA_ENV"

export HF_HOME=/scratch/$USER/hf_cache
export TRANSFORMERS_CACHE=/scratch/$USER/hf_cache
export XDG_CACHE_HOME=/scratch/$USER/.cache

echo "Host: $(hostname)"
echo "Python: $(which python)"
echo "OLMOCR: $(which olmocr)"
echo "PDFTOPPM: $(which pdftoppm || true)"
nvidia-smi

python olmocr_client.py \
  --input_dir "$INPUT_DIR" \
  --workspace "$WORKSPACE" \
  --chunk_size "$CHUNK_SIZE" \
  --workers "$WORKERS"
