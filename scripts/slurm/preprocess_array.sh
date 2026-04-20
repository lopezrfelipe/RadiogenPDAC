#!/bin/bash
# Example SLURM array wrapper for preprocessing and cohort QC.
# Replace module/conda activation with your site defaults.

#SBATCH --job-name=pdac-preprocess
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --array=0-4
#SBATCH --output=logs/preprocess_%A_%a.out

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$PWD}"
COHORT_MANIFEST="${COHORT_MANIFEST:-${PROJECT_ROOT}/templates/cohort_manifest.example.csv}"
GENOMICS_MANIFEST="${GENOMICS_MANIFEST:-${PROJECT_ROOT}/templates/genomics_manifest.example.csv}"

cd "${PROJECT_ROOT}"

python -m radiogenpdac.cli validate-manifest \
  --cohort "${COHORT_MANIFEST}" \
  --genomics "${GENOMICS_MANIFEST}"

python -m radiogenpdac.cli merge-manifests \
  --cohort "${COHORT_MANIFEST}" \
  --genomics "${GENOMICS_MANIFEST}" \
  --output "${PROJECT_ROOT}/artifacts/training_manifest.csv"

python -m radiogenpdac.cli make-splits \
  --manifest "${PROJECT_ROOT}/artifacts/training_manifest.csv" \
  --output "${PROJECT_ROOT}/artifacts/split_assignment.csv" \
  --n-folds 5 \
  --group-column split_group \
  --stratify-column site \
  --test-fraction 0.15 \
  --seed 2026

python -m radiogenpdac.cli preprocess-cohort \
  --manifest "${PROJECT_ROOT}/artifacts/training_manifest.csv" \
  --data-config "${PROJECT_ROOT}/configs/data/default.yaml" \
  --model-config "${PROJECT_ROOT}/configs/model/multiphase_encoder.yaml" \
  --output-dir "${PROJECT_ROOT}/artifacts/preprocessed" \
  --output-manifest "${PROJECT_ROOT}/artifacts/preprocessed_manifest.csv"
