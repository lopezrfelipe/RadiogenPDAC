#!/bin/bash
# Rebuild the venous fine-tuning dataset from the hybrid-vein manifest using
# the fixed multiclass overlap handling in ingestion.py, then write splits and run
# nnU-Net planning/preprocessing.
#
# Submit with:
#   sbatch scripts/slurm/venous_full_hybrid_vein_prep.sbatch
#
# Optional overrides:
#   PROJECT_ROOT=/path/to/RadiogenPDAC
#   CONDA_ENV_NAME=pdac-ft
#   NNUNET_RAW_DIR=/path/to/nnUNet_raw
#   NNUNET_PREPROCESSED_DIR=/path/to/nnUNet_preprocessed
#   NNUNET_RESULTS_DIR=/path/to/nnUNet_results
#   PANDX_WORKSPACE=/path/to/pandx_workspace
#   DATA_ROOT=/path/to/data
#   PHASE_MANIFEST=/path/to/venous_training_manifest.hybrid_vessels.csv
#   DATASET_ID=915
#   DATASET_NAME=PDACVenousFullHybridVessels
#   PLANS_IDENTIFIER=nnUNetPlansFrom107v3
#   CONFIGURATION=3d_fullres
#   N_FOLDS=5
#   SEED=12345

#SBATCH --job-name=pdac-ven-hybrid-vess-prep
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --output=logs/venous_full_hybrid_vessels_prep_%j.out

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$PWD}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-pdac-ft}"
REPO_PARENT="$(cd "${PROJECT_ROOT}/.." && pwd)"
NNUNET_RAW_DIR="${NNUNET_RAW_DIR:-${REPO_PARENT}/nnUNet_raw}"
NNUNET_PREPROCESSED_DIR="${NNUNET_PREPROCESSED_DIR:-${REPO_PARENT}/nnUNet_preprocessed}"
NNUNET_RESULTS_DIR="${NNUNET_RESULTS_DIR:-${REPO_PARENT}/nnUNet_results}"
PANDX_WORKSPACE="${PANDX_WORKSPACE:-${REPO_PARENT}/pandx_workspace}"
DATA_ROOT="${DATA_ROOT:-${REPO_PARENT}/data}"
PHASE_MANIFEST="${PHASE_MANIFEST:-${DATA_ROOT}/manifests/venous_full/venous_training_manifest.hybrid_vessels.csv}"
PREPARED_INDEX="${PREPARED_INDEX:-${DATA_ROOT}/manifests/venous_full/venous_hybrid_vessels_prepared_index.csv}"
DATASET_ID="${DATASET_ID:-915}"
DATASET_NAME="${DATASET_NAME:-PDACVenousFullHybridVessels}"
PLANS_IDENTIFIER="${PLANS_IDENTIFIER:-nnUNetPlansFrom107v3}"
CONFIGURATION="${CONFIGURATION:-3d_fullres}"
GPU_MEMORY_TARGET_GB="${GPU_MEMORY_TARGET_GB:-8.0}"
NUM_PROCESSES="${NUM_PROCESSES:-4}"
N_FOLDS="${N_FOLDS:-5}"
SEED="${SEED:-12345}"
ORIGINAL_MODEL_TRAINING_OUTPUT_DIR="${ORIGINAL_MODEL_TRAINING_OUTPUT_DIR:-${PANDX_WORKSPACE}/nnUNet_results/Dataset107_PDAC_Detection/nnUNetTrainerCELossLesionSplit__nnUNetPlans_v3__3d_fullres}"
PDAC_ROOT="${PDAC_ROOT:-${PROJECT_ROOT}/PDAC_Detection}"
PYTHON_BIN="${PYTHON_BIN:-/home/felipe/.conda/envs/${CONDA_ENV_NAME}/bin/python}"

mkdir -p "${PROJECT_ROOT}/logs"

if [[ ! -f "${PHASE_MANIFEST}" ]]; then
  echo "Missing hybrid phase manifest: ${PHASE_MANIFEST}"
  exit 1
fi

if [[ ! -d "${ORIGINAL_MODEL_TRAINING_OUTPUT_DIR}" ]]; then
  echo "Missing original model training output dir: ${ORIGINAL_MODEL_TRAINING_OUTPUT_DIR}"
  exit 1
fi

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Missing explicit python binary: ${PYTHON_BIN}"
  exit 1
fi

cd "${PROJECT_ROOT}"

module load miniconda/3.13
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV_NAME}"

export PATH="/home/felipe/.conda/envs/${CONDA_ENV_NAME}/bin:${PATH}"
export PYTHONPATH="${PROJECT_ROOT}/src"
hash -r

"${PYTHON_BIN}" -V

echo "[prep] rebuilding venous hybrid-vein raw dataset"
"${PYTHON_BIN}" -m radiogenpdac.cli \
  prepare-ingested-encoder-dataset \
  --phase-manifest "${PHASE_MANIFEST}" \
  --phase venous \
  --dataset-id "${DATASET_ID}" \
  --dataset-name "${DATASET_NAME}" \
  --pdac-root "${PDAC_ROOT}" \
  --nnunet-raw-dir "${NNUNET_RAW_DIR}" \
  --output-index "${PREPARED_INDEX}" \
  --task-mode multiclass \
  --crop-mode pancreas_roi \
  --structure-priority pancreas,tumor,artery,vein,cbd,duct,cyst

echo "[prep] writing encoder splits"
"${PYTHON_BIN}" -m radiogenpdac.cli \
  write-encoder-splits \
  --prepared-index "${PREPARED_INDEX}" \
  --nnunet-preprocessed-dir "${NNUNET_PREPROCESSED_DIR}" \
  --dataset-id "${DATASET_ID}" \
  --dataset-name "${DATASET_NAME}" \
  --n-folds "${N_FOLDS}" \
  --seed "${SEED}"

echo "[prep] planning and preprocessing hybrid venous dataset"
"${PYTHON_BIN}" -m radiogenpdac.cli \
  plan-encoder-dataset \
  --dataset-id "${DATASET_ID}" \
  --pdac-root "${PDAC_ROOT}" \
  --nnunet-raw-dir "${NNUNET_RAW_DIR}" \
  --nnunet-preprocessed-dir "${NNUNET_PREPROCESSED_DIR}" \
  --nnunet-results-dir "${NNUNET_RESULTS_DIR}" \
  --plans-identifier "${PLANS_IDENTIFIER}" \
  --configurations "${CONFIGURATION}" \
  --gpu-memory-target-gb "${GPU_MEMORY_TARGET_GB}" \
  --num-processes "${NUM_PROCESSES}" \
  --source-model-training-output-dir "${ORIGINAL_MODEL_TRAINING_OUTPUT_DIR}"
