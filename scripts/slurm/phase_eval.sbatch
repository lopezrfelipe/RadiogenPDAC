#!/bin/bash
# Generic tumor evaluation for the arterial or venous phase on any fold.
#
# Submit with:
#   PHASE=arterial EVAL_TARGET=baseline FOLD=0 sbatch scripts/slurm/phase_eval.sbatch
#   PHASE=arterial EVAL_TARGET=best FOLD=1 sbatch scripts/slurm/phase_eval.sbatch
#   PHASE=venous EVAL_TARGET=final FOLD=0 sbatch scripts/slurm/phase_eval.sbatch
#
# Supported targets:
#   baseline -> original pretrained model, checkpoint_final.pth
#   best     -> fine-tuned model, checkpoint_best.pth
#   latest   -> fine-tuned model, checkpoint_latest.pth
#   last     -> alias for latest
#   final    -> fine-tuned model, checkpoint_final.pth
#
# Optional overrides:
#   PROJECT_ROOT=/path/to/RadiogenPDAC
#   CONDA_ENV_NAME=pdac-ft
#   PANDX_WORKSPACE=/path/to/pandx_workspace
#   NNUNET_RAW_DIR=/path/to/nnUNet_raw
#   NNUNET_PREPROCESSED_DIR=/path/to/nnUNet_preprocessed
#   NNUNET_RESULTS_DIR=/path/to/nnUNet_results
#   DATA_ROOT=/path/to/data
#   PHASE=venous|arterial
#   EVAL_TARGET=baseline|best|latest|last|final
#   DATASET_ID=912
#   DATASET_NAME=PDACVenousFullHybridArtery
#   DATASET_OUTPUT_SUBDIR=venous_full
#   TRAINER_NAME=nnUNetTrainer_ftce
#   PLANS_IDENTIFIER=nnUNetPlans_v3
#   CONFIGURATION=3d_fullres
#   FOLD=0
#   REFERENCE_TUMOR_LABEL=1
#   PREDICTION_TUMOR_LABEL=1
#   BASELINE_MODEL_TRAINING_OUTPUT_DIR=/path/to/original/model_training_output_dir
#   FINETUNED_MODEL_TRAINING_OUTPUT_DIR=/path/to/finetuned/model_training_output_dir
#   MODEL_TRAINING_OUTPUT_DIR=/path/to/model_training_output_dir
#   CHECKPOINT_NAME=checkpoint_best.pth
#   OUTPUT_DIR=/path/to/eval_output_dir
#
# Example:
#   PHASE=venous EVAL_TARGET=best DATASET_ID=912 DATASET_NAME=PDACVenousFullHybridArtery \
#   FOLD=0 REFERENCE_TUMOR_LABEL=1 PREDICTION_TUMOR_LABEL=1 \
#   sbatch scripts/slurm/phase_eval.sbatch

#SBATCH --job-name=pdac-phase-eval
#SBATCH --partition=gpu,gpx
#SBATCH --nodelist=felixgpu01,felixgpx03
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=logs/phase_eval_%j.out

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$PWD}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-pdac-ft}"
PANDX_WORKSPACE="${PANDX_WORKSPACE:-$(cd "${PROJECT_ROOT}/.." && pwd)/pandx_workspace}"
NNUNET_RAW_DIR="${NNUNET_RAW_DIR:-$(cd "${PROJECT_ROOT}/.." && pwd)/nnUNet_raw}"
NNUNET_PREPROCESSED_DIR="${NNUNET_PREPROCESSED_DIR:-$(cd "${PROJECT_ROOT}/.." && pwd)/nnUNet_preprocessed}"
NNUNET_RESULTS_DIR="${NNUNET_RESULTS_DIR:-$(cd "${PROJECT_ROOT}/.." && pwd)/nnUNet_results}"
DATA_ROOT="${DATA_ROOT:-$(cd "${PROJECT_ROOT}/.." && pwd)/data}"
PHASE="${PHASE:-arterial}"
EVAL_TARGET="${EVAL_TARGET:-baseline}"
TRAINER_NAME="${TRAINER_NAME:-nnUNetTrainer_ftce}"
PLANS_IDENTIFIER="${PLANS_IDENTIFIER:-nnUNetPlans_v3}"
CONFIGURATION="${CONFIGURATION:-3d_fullres}"
FOLD="${FOLD:-0}"
PYTHON_BIN="${PYTHON_BIN:-/home/felipe/.conda/envs/${CONDA_ENV_NAME}/bin/python}"
DATASET_ID="${DATASET_ID:-}"
DATASET_NAME="${DATASET_NAME:-}"
DATASET_OUTPUT_SUBDIR="${DATASET_OUTPUT_SUBDIR:-}"
REFERENCE_TUMOR_LABEL="${REFERENCE_TUMOR_LABEL:-}"
PREDICTION_TUMOR_LABEL="${PREDICTION_TUMOR_LABEL:-}"

case "${PHASE}" in
  arterial)
    DEFAULT_DATASET_OUTPUT_SUBDIR="arterial_250"
    ;;
  venous)
    DEFAULT_DATASET_OUTPUT_SUBDIR="venous_full"
    ;;
  *)
    echo "Unsupported PHASE: ${PHASE}. Expected arterial or venous."
    exit 1
    ;;
esac

if [[ -z "${DATASET_ID}" ]]; then
  echo "DATASET_ID is required."
  exit 1
fi

if [[ -z "${DATASET_NAME}" ]]; then
  echo "DATASET_NAME is required."
  exit 1
fi

if [[ -z "${REFERENCE_TUMOR_LABEL}" ]]; then
  echo "REFERENCE_TUMOR_LABEL is required."
  exit 1
fi

if [[ -z "${PREDICTION_TUMOR_LABEL}" ]]; then
  echo "PREDICTION_TUMOR_LABEL is required."
  exit 1
fi

DATASET_OUTPUT_SUBDIR="${DATASET_OUTPUT_SUBDIR:-${DEFAULT_DATASET_OUTPUT_SUBDIR}}"

BASELINE_MODEL_TRAINING_OUTPUT_DIR="${BASELINE_MODEL_TRAINING_OUTPUT_DIR:-${PANDX_WORKSPACE}/nnUNet_results/Dataset107_PDAC_Detection/nnUNetTrainerCELossLesionSplit__nnUNetPlans_v3__3d_fullres}"
FINETUNED_MODEL_TRAINING_OUTPUT_DIR="${FINETUNED_MODEL_TRAINING_OUTPUT_DIR:-${NNUNET_RESULTS_DIR}/Dataset$(printf "%03d" "${DATASET_ID}")_${DATASET_NAME}/${TRAINER_NAME}__${PLANS_IDENTIFIER}__${CONFIGURATION}}"

case "${EVAL_TARGET}" in
  baseline)
    DEFAULT_MODEL_TRAINING_OUTPUT_DIR="${BASELINE_MODEL_TRAINING_OUTPUT_DIR}"
    DEFAULT_CHECKPOINT_NAME="checkpoint_final.pth"
    DEFAULT_OUTPUT_SUFFIX="baseline_eval"
    ;;
  best)
    DEFAULT_MODEL_TRAINING_OUTPUT_DIR="${FINETUNED_MODEL_TRAINING_OUTPUT_DIR}"
    DEFAULT_CHECKPOINT_NAME="checkpoint_best.pth"
    DEFAULT_OUTPUT_SUFFIX="ft_eval_best"
    ;;
  latest|last)
    DEFAULT_MODEL_TRAINING_OUTPUT_DIR="${FINETUNED_MODEL_TRAINING_OUTPUT_DIR}"
    DEFAULT_CHECKPOINT_NAME="checkpoint_latest.pth"
    DEFAULT_OUTPUT_SUFFIX="ft_eval_latest"
    ;;
  final)
    DEFAULT_MODEL_TRAINING_OUTPUT_DIR="${FINETUNED_MODEL_TRAINING_OUTPUT_DIR}"
    DEFAULT_CHECKPOINT_NAME="checkpoint_final.pth"
    DEFAULT_OUTPUT_SUFFIX="ft_eval_final"
    ;;
  *)
    echo "Unsupported EVAL_TARGET: ${EVAL_TARGET}. Expected baseline, best, latest, last, or final."
    exit 1
    ;;
esac

MODEL_TRAINING_OUTPUT_DIR="${MODEL_TRAINING_OUTPUT_DIR:-${DEFAULT_MODEL_TRAINING_OUTPUT_DIR}}"
CHECKPOINT_NAME="${CHECKPOINT_NAME:-${DEFAULT_CHECKPOINT_NAME}}"
OUTPUT_DIR="${OUTPUT_DIR:-${DATA_ROOT}/manifests/${DATASET_OUTPUT_SUBDIR}/fold_${FOLD}_${DEFAULT_OUTPUT_SUFFIX}}"

DATASET_DIR="${NNUNET_RAW_DIR}/Dataset$(printf "%03d" "${DATASET_ID}")_${DATASET_NAME}"
PREPROCESSED_DATASET_DIR="${NNUNET_PREPROCESSED_DIR}/Dataset$(printf "%03d" "${DATASET_ID}")_${DATASET_NAME}"
CHECKPOINT_PATH="${MODEL_TRAINING_OUTPUT_DIR}/fold_${FOLD}/${CHECKPOINT_NAME}"

mkdir -p "${PROJECT_ROOT}/logs" "${OUTPUT_DIR}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Missing explicit python binary: ${PYTHON_BIN}"
  exit 1
fi

if [[ ! -d "${MODEL_TRAINING_OUTPUT_DIR}" ]]; then
  echo "Missing model training output dir: ${MODEL_TRAINING_OUTPUT_DIR}"
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
  --reference-tumor-label "${REFERENCE_TUMOR_LABEL}" \
  --prediction-tumor-label "${PREDICTION_TUMOR_LABEL}" \
  --split-json "${PREPROCESSED_DATASET_DIR}/splits_final.json" \
  --fold "${FOLD}" \
  --checkpoint-name "${CHECKPOINT_NAME}" \
  --device cuda
