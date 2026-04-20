#!/bin/bash
# Full venous baseline evaluation on fold 0 after dataset prep/preprocessing is complete.
# Submit with:
#   sbatch scripts/slurm/venous_full_fold0_baseline.sbatch
#
# Expected pre-existing assets:
#   - nnUNet raw dataset for Dataset911_PDACVenousFull
#   - nnUNet preprocessed metadata for Dataset911_PDACVenousFull
#   - splits_final.json
#
# Optional overrides:
#   PROJECT_ROOT=/path/to/RadiogenPDAC
#   CONDA_ENV_NAME=pdac-ft
#   PANDX_WORKSPACE=/path/to/pandx_workspace
#   NNUNET_RAW_DIR=/path/to/nnUNet_raw
#   NNUNET_PREPROCESSED_DIR=/path/to/nnUNet_preprocessed
#   NNUNET_RESULTS_DIR=/path/to/nnUNet_results
#   DATA_ROOT=/path/to/data
#   DATASET_ID=911
#   DATASET_NAME=PDACVenousFull
#   FOLD=0
#   ORIGINAL_MODEL_TRAINING_OUTPUT_DIR=/path/to/original/model_training_output_dir

#SBATCH --job-name=pdac-ven-full-base0
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=logs/venous_full_baseline0_%j.out

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$PWD}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-pdac-ft}"
PANDX_WORKSPACE="${PANDX_WORKSPACE:-$(cd "${PROJECT_ROOT}/.." && pwd)/pandx_workspace}"
NNUNET_RAW_DIR="${NNUNET_RAW_DIR:-$(cd "${PROJECT_ROOT}/.." && pwd)/nnUNet_raw}"
NNUNET_PREPROCESSED_DIR="${NNUNET_PREPROCESSED_DIR:-$(cd "${PROJECT_ROOT}/.." && pwd)/nnUNet_preprocessed}"
NNUNET_RESULTS_DIR="${NNUNET_RESULTS_DIR:-$(cd "${PROJECT_ROOT}/.." && pwd)/nnUNet_results}"
DATA_ROOT="${DATA_ROOT:-$(cd "${PROJECT_ROOT}/.." && pwd)/data}"
DATASET_ID="${DATASET_ID:-911}"
DATASET_NAME="${DATASET_NAME:-PDACVenousFull}"
FOLD="${FOLD:-0}"
ORIGINAL_MODEL_TRAINING_OUTPUT_DIR="${ORIGINAL_MODEL_TRAINING_OUTPUT_DIR:-${PANDX_WORKSPACE}/nnUNet_results/Dataset107_PDAC_Detection/nnUNetTrainerCELossLesionSplit__nnUNetPlans_v3__3d_fullres}"

DATASET_DIR="${NNUNET_RAW_DIR}/Dataset$(printf "%03d" "${DATASET_ID}")_${DATASET_NAME}"
PREPROCESSED_DATASET_DIR="${NNUNET_PREPROCESSED_DIR}/Dataset$(printf "%03d" "${DATASET_ID}")_${DATASET_NAME}"
OUTPUT_DIR="${DATA_ROOT}/manifests/venous_full/fold_${FOLD}_baseline_eval"

mkdir -p "${PROJECT_ROOT}/logs" "${OUTPUT_DIR}"

if [[ ! -d "${ORIGINAL_MODEL_TRAINING_OUTPUT_DIR}" ]]; then
  echo "Missing original model training output dir: ${ORIGINAL_MODEL_TRAINING_OUTPUT_DIR}"
  exit 1
fi

if [[ ! -f "${PREPROCESSED_DATASET_DIR}/splits_final.json" ]]; then
  echo "Missing splits file: ${PREPROCESSED_DATASET_DIR}/splits_final.json"
  exit 1
fi

if [[ ! -d "${DATASET_DIR}/imagesTr" ]]; then
  echo "Missing raw images folder: ${DATASET_DIR}/imagesTr"
  exit 1
fi

if [[ ! -d "${DATASET_DIR}/labelsTr" ]]; then
  echo "Missing raw labels folder: ${DATASET_DIR}/labelsTr"
  exit 1
fi

cd "${PROJECT_ROOT}"

module load miniconda/3.13
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV_NAME}"

radiogenpdac evaluate-encoder-model \
  --pdac-root "${PROJECT_ROOT}/PDAC_Detection" \
  --nnunet-raw-dir "${NNUNET_RAW_DIR}" \
  --nnunet-preprocessed-dir "${NNUNET_PREPROCESSED_DIR}" \
  --nnunet-results-dir "${NNUNET_RESULTS_DIR}" \
  --model-training-output-dir "${ORIGINAL_MODEL_TRAINING_OUTPUT_DIR}" \
  --images-folder "${DATASET_DIR}/imagesTr" \
  --reference-folder "${DATASET_DIR}/labelsTr" \
  --output-folder "${OUTPUT_DIR}" \
  --reference-tumor-label 2 \
  --prediction-tumor-label 1 \
  --split-json "${PREPROCESSED_DATASET_DIR}/splits_final.json" \
  --fold "${FOLD}" \
  --checkpoint-name checkpoint_final.pth \
  --device cuda
