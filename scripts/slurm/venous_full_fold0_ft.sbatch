#!/bin/bash
# Full venous multiclass fine-tuning on fold 0 after dataset prep/preprocessing is complete.
# Submit with:
#   sbatch scripts/slurm/venous_full_fold0_ft.sbatch
#
# Expected pre-existing assets:
#   - nnUNet raw dataset for Dataset911_PDACVenousFull
#   - nnUNet preprocessed metadata for Dataset911_PDACVenousFull
#   - splits_final.json
#   - nnUNetPlansFrom107v3.json
#   - dataset_fingerprint.json
#
# Optional overrides:
#   PROJECT_ROOT=/path/to/RadiogenPDAC
#   CONDA_ENV_NAME=pdac-ft
#   PANDX_WORKSPACE=/path/to/pandx_workspace
#   NNUNET_RAW_DIR=/path/to/nnUNet_raw
#   NNUNET_PREPROCESSED_DIR=/path/to/nnUNet_preprocessed
#   NNUNET_RESULTS_DIR=/path/to/nnUNet_results
#   DATASET_ID=911
#   DATASET_NAME=PDACVenousFull
#   TRAINER_NAME=nnUNetTrainer_ftce
#   PLANS_IDENTIFIER=nnUNetPlansFrom107v3
#   CONFIGURATION=3d_fullres
#   FOLD=0
#   ORIGINAL_MODEL_TRAINING_OUTPUT_DIR=/path/to/original/model_training_output_dir
#   PRETRAINED_WEIGHTS=/path/to/checkpoint_final.pth

#SBATCH --job-name=pdac-ven-full-ft0
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=48:00:00
#SBATCH --output=logs/venous_full_ft0_%j.out

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$PWD}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-pdac-ft}"
PANDX_WORKSPACE="${PANDX_WORKSPACE:-$(cd "${PROJECT_ROOT}/.." && pwd)/pandx_workspace}"
NNUNET_RAW_DIR="${NNUNET_RAW_DIR:-$(cd "${PROJECT_ROOT}/.." && pwd)/nnUNet_raw}"
NNUNET_PREPROCESSED_DIR="${NNUNET_PREPROCESSED_DIR:-$(cd "${PROJECT_ROOT}/.." && pwd)/nnUNet_preprocessed}"
NNUNET_RESULTS_DIR="${NNUNET_RESULTS_DIR:-$(cd "${PROJECT_ROOT}/.." && pwd)/nnUNet_results}"
DATASET_ID="${DATASET_ID:-911}"
DATASET_NAME="${DATASET_NAME:-PDACVenousFull}"
TRAINER_NAME="${TRAINER_NAME:-nnUNetTrainer_ftce}"
PLANS_IDENTIFIER="${PLANS_IDENTIFIER:-nnUNetPlansFrom107v3}"
CONFIGURATION="${CONFIGURATION:-3d_fullres}"
FOLD="${FOLD:-0}"
ORIGINAL_MODEL_TRAINING_OUTPUT_DIR="${ORIGINAL_MODEL_TRAINING_OUTPUT_DIR:-${PANDX_WORKSPACE}/nnUNet_results/Dataset107_PDAC_Detection/nnUNetTrainerCELossLesionSplit__nnUNetPlans_v3__3d_fullres}"
PRETRAINED_WEIGHTS="${PRETRAINED_WEIGHTS:-${ORIGINAL_MODEL_TRAINING_OUTPUT_DIR}/fold_0/checkpoint_final.pth}"

DATASET_DIR="${NNUNET_RAW_DIR}/Dataset$(printf "%03d" "${DATASET_ID}")_${DATASET_NAME}"
PREPROCESSED_DATASET_DIR="${NNUNET_PREPROCESSED_DIR}/Dataset$(printf "%03d" "${DATASET_ID}")_${DATASET_NAME}"

mkdir -p "${PROJECT_ROOT}/logs"

if [[ ! -f "${PRETRAINED_WEIGHTS}" ]]; then
  echo "Missing pretrained weights: ${PRETRAINED_WEIGHTS}"
  exit 1
fi

if [[ ! -f "${PREPROCESSED_DATASET_DIR}/splits_final.json" ]]; then
  echo "Missing splits file: ${PREPROCESSED_DATASET_DIR}/splits_final.json"
  exit 1
fi

if [[ ! -f "${PREPROCESSED_DATASET_DIR}/${PLANS_IDENTIFIER}.json" ]]; then
  echo "Missing plans file: ${PREPROCESSED_DATASET_DIR}/${PLANS_IDENTIFIER}.json"
  exit 1
fi

if [[ ! -f "${PREPROCESSED_DATASET_DIR}/dataset_fingerprint.json" ]]; then
  echo "Missing dataset fingerprint: ${PREPROCESSED_DATASET_DIR}/dataset_fingerprint.json"
  exit 1
fi

if [[ ! -f "${DATASET_DIR}/dataset.json" ]]; then
  echo "Missing raw dataset json: ${DATASET_DIR}/dataset.json"
  exit 1
fi

cd "${PROJECT_ROOT}"

module load miniconda/3.13
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV_NAME}"

radiogenpdac finetune-encoder \
  --dataset-id "${DATASET_ID}" \
  --pdac-root "${PROJECT_ROOT}/PDAC_Detection" \
  --nnunet-raw-dir "${NNUNET_RAW_DIR}" \
  --nnunet-preprocessed-dir "${NNUNET_PREPROCESSED_DIR}" \
  --nnunet-results-dir "${NNUNET_RESULTS_DIR}" \
  --pretrained-weights "${PRETRAINED_WEIGHTS}" \
  --configuration "${CONFIGURATION}" \
  --fold "${FOLD}" \
  --trainer-name "${TRAINER_NAME}" \
  --plans-identifier "${PLANS_IDENTIFIER}" \
  --device cuda \
  --num-gpus 1
