#!/bin/bash
# Example multi-GPU training launcher template.
# The actual Lightning/MONAI training loop should consume the same configs
# defined under configs/ once model code is added.

#SBATCH --job-name=pdac-train
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=48:00:00
#SBATCH --output=logs/train_%j.out

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$PWD}"
mkdir -p "${PROJECT_ROOT}/logs" "${PROJECT_ROOT}/artifacts"

cd "${PROJECT_ROOT}"

python -m radiogenpdac.cli train \
  --manifest "${PROJECT_ROOT}/artifacts/preprocessed_manifest.csv" \
  --splits "${PROJECT_ROOT}/artifacts/split_assignment.csv" \
  --fold 0 \
  --data-config "${PROJECT_ROOT}/configs/data/default.yaml" \
  --model-config "${PROJECT_ROOT}/configs/model/multiphase_encoder.yaml" \
  --target-config "${PROJECT_ROOT}/configs/target/hybrid_signature.yaml" \
  --train-config "${PROJECT_ROOT}/configs/train/default.yaml" \
  --output-dir "${PROJECT_ROOT}/artifacts/runs/fold0"
