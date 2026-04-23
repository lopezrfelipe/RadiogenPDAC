#!/bin/bash
# Arterial 250 fine-tuned tumor evaluation on fold 0 using the best checkpoint so far.
#
# Submit with:
#   sbatch scripts/slurm/arterial_250_fold0_best_eval.sbatch
#
# Expected pre-existing assets:
#   - nnUNet raw dataset for Dataset913_PDACArterial250
#   - nnUNet preprocessed metadata for Dataset913_PDACArterial250
#   - splits_final.json
#   - fine-tuned model outputs for Dataset913_PDACArterial250
#
# Optional overrides:
#   PROJECT_ROOT=/path/to/RadiogenPDAC
#   CONDA_ENV_NAME=pdac-ft
#   NNUNET_RAW_DIR=/path/to/nnUNet_raw
#   NNUNET_PREPROCESSED_DIR=/path/to/nnUNet_preprocessed
#   NNUNET_RESULTS_DIR=/path/to/nnUNet_results
#   DATA_ROOT=/path/to/data
#   DATASET_ID=913
#   DATASET_NAME=PDACArterial250
#   TRAINER_NAME=nnUNetTrainer_ftce
#   PLANS_IDENTIFIER=nnUNetPlansFrom107v3
#   CONFIGURATION=3d_fullres
#   FOLD=0
#   MODEL_TRAINING_OUTPUT_DIR=/path/to/finetuned/model_training_output_dir
#   CHECKPOINT_NAME=checkpoint_best.pth
#   OUTPUT_DIR=/path/to/eval_output_dir

#SBATCH --job-name=pdac-art250-best0
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=logs/arterial_250_best0_%j.out

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$PWD}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-pdac-ft}"
NNUNET_RAW_DIR="${NNUNET_RAW_DIR:-$(cd "${PROJECT_ROOT}/.." && pwd)/nnUNet_raw}"
NNUNET_PREPROCESSED_DIR="${NNUNET_PREPROCESSED_DIR:-$(cd "${PROJECT_ROOT}/.." && pwd)/nnUNet_preprocessed}"
NNUNET_RESULTS_DIR="${NNUNET_RESULTS_DIR:-$(cd "${PROJECT_ROOT}/.." && pwd)/nnUNet_results}"
DATA_ROOT="${DATA_ROOT:-$(cd "${PROJECT_ROOT}/.." && pwd)/data}"
DATASET_ID="${DATASET_ID:-913}"
DATASET_NAME="${DATASET_NAME:-PDACArterial250}"
TRAINER_NAME="${TRAINER_NAME:-nnUNetTrainer_ftce}"
PLANS_IDENTIFIER="${PLANS_IDENTIFIER:-nnUNetPlansFrom107v3}"
CONFIGURATION="${CONFIGURATION:-3d_fullres}"
FOLD="${FOLD:-0}"
MODEL_TRAINING_OUTPUT_DIR="${MODEL_TRAINING_OUTPUT_DIR:-${NNUNET_RESULTS_DIR}/Dataset$(printf "%03d" "${DATASET_ID}")_${DATASET_NAME}/${TRAINER_NAME}__${PLANS_IDENTIFIER}__${CONFIGURATION}}"
CHECKPOINT_NAME="${CHECKPOINT_NAME:-checkpoint_best.pth}"
PYTHON_BIN="${PYTHON_BIN:-/home/felipe/.conda/envs/${CONDA_ENV_NAME}/bin/python}"

DATASET_DIR="${NNUNET_RAW_DIR}/Dataset$(printf "%03d" "${DATASET_ID}")_${DATASET_NAME}"
PREPROCESSED_DATASET_DIR="${NNUNET_PREPROCESSED_DIR}/Dataset$(printf "%03d" "${DATASET_ID}")_${DATASET_NAME}"
OUTPUT_DIR="${OUTPUT_DIR:-${DATA_ROOT}/manifests/arterial_250/fold_${FOLD}_ft_eval_best}"
CHECKPOINT_PATH="${MODEL_TRAINING_OUTPUT_DIR}/fold_${FOLD}/${CHECKPOINT_NAME}"

mkdir -p "${PROJECT_ROOT}/logs" "${OUTPUT_DIR}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Missing explicit python binary: ${PYTHON_BIN}"
  exit 1
fi

if [[ ! -d "${MODEL_TRAINING_OUTPUT_DIR}" ]]; then
  echo "Missing fine-tuned model training output dir: ${MODEL_TRAINING_OUTPUT_DIR}"
  exit 1
fi

if [[ ! -f "${CHECKPOINT_PATH}" ]]; then
  echo "Missing checkpoint: ${CHECKPOINT_PATH}"
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

export PATH="/home/felipe/.conda/envs/${CONDA_ENV_NAME}/bin:${PATH}"
export PYTHONPATH="${PROJECT_ROOT}/src"
hash -r

"${PYTHON_BIN}" -V

"${PYTHON_BIN}" -m radiogenpdac.cli \
  evaluate-encoder-model \
  --pdac-root "${PROJECT_ROOT}/PDAC_Detection" \
  --nnunet-raw-dir "${NNUNET_RAW_DIR}" \
  --nnunet-preprocessed-dir "${NNUNET_PREPROCESSED_DIR}" \
  --nnunet-results-dir "${NNUNET_RESULTS_DIR}" \
  --model-training-output-dir "${MODEL_TRAINING_OUTPUT_DIR}" \
  --images-folder "${DATASET_DIR}/imagesTr" \
  --reference-folder "${DATASET_DIR}/labelsTr" \
  --output-folder "${OUTPUT_DIR}" \
  --reference-tumor-label 1 \
  --prediction-tumor-label 1 \
  --split-json "${PREPROCESSED_DATASET_DIR}/splits_final.json" \
  --fold "${FOLD}" \
  --checkpoint-name "${CHECKPOINT_NAME}" \
  --device cuda
