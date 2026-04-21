#!/bin/bash
# Poll a nnU-Net training monitor CSV and submit validation-eval jobs every N epochs.
# Intended to run from a login shell without reserving GPUs between evaluations.
#
# Required environment variables:
#   PROJECT_ROOT=/path/to/RadiogenPDAC
#   MODEL_TRAINING_OUTPUT_DIR=/path/to/model/output/dir
#   NNUNET_RAW_DIR=/path/to/nnUNet_raw
#   NNUNET_PREPROCESSED_DIR=/path/to/nnUNet_preprocessed
#   NNUNET_RESULTS_DIR=/path/to/nnUNet_results
#   IMAGES_FOLDER=/path/to/imagesTr
#   REFERENCE_FOLDER=/path/to/labelsTr
#   SPLIT_JSON=/path/to/splits_final.json
#   OUTPUT_ROOT=/path/to/eval/output/root
#
# Optional environment variables:
#   CONDA_ENV_NAME=pdac-ft
#   PDAC_ROOT=/path/to/PDAC_Detection
#   FOLD=0
#   EPOCH_STEP=50
#   POLL_INTERVAL_SEC=300
#   REFERENCE_TUMOR_LABEL=2
#   PREDICTION_TUMOR_LABEL=2
#   CHECKPOINT_NAME=checkpoint_latest.pth

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-}"
MODEL_TRAINING_OUTPUT_DIR="${MODEL_TRAINING_OUTPUT_DIR:-}"
NNUNET_RAW_DIR="${NNUNET_RAW_DIR:-}"
NNUNET_PREPROCESSED_DIR="${NNUNET_PREPROCESSED_DIR:-}"
NNUNET_RESULTS_DIR="${NNUNET_RESULTS_DIR:-}"
IMAGES_FOLDER="${IMAGES_FOLDER:-}"
REFERENCE_FOLDER="${REFERENCE_FOLDER:-}"
SPLIT_JSON="${SPLIT_JSON:-}"
OUTPUT_ROOT="${OUTPUT_ROOT:-}"

CONDA_ENV_NAME="${CONDA_ENV_NAME:-pdac-ft}"
PDAC_ROOT="${PDAC_ROOT:-${PROJECT_ROOT}/PDAC_Detection}"
FOLD="${FOLD:-0}"
EPOCH_STEP="${EPOCH_STEP:-50}"
POLL_INTERVAL_SEC="${POLL_INTERVAL_SEC:-300}"
REFERENCE_TUMOR_LABEL="${REFERENCE_TUMOR_LABEL:-2}"
PREDICTION_TUMOR_LABEL="${PREDICTION_TUMOR_LABEL:-2}"
CHECKPOINT_NAME="${CHECKPOINT_NAME:-checkpoint_latest.pth}"

for required_name in PROJECT_ROOT MODEL_TRAINING_OUTPUT_DIR NNUNET_RAW_DIR NNUNET_PREPROCESSED_DIR NNUNET_RESULTS_DIR IMAGES_FOLDER REFERENCE_FOLDER SPLIT_JSON OUTPUT_ROOT; do
  if [[ -z "${!required_name}" ]]; then
    echo "Missing required environment variable: ${required_name}"
    exit 1
  fi
done

MONITOR_CSV="${MODEL_TRAINING_OUTPUT_DIR}/fold_${FOLD}/training_monitor.csv"
CHECKPOINT_FINAL="${MODEL_TRAINING_OUTPUT_DIR}/fold_${FOLD}/checkpoint_final.pth"
STATE_DIR="${OUTPUT_ROOT}/.submitted_eval_epochs"
mkdir -p "${STATE_DIR}" "${OUTPUT_ROOT}" "${PROJECT_ROOT}/logs"

next_epoch="${EPOCH_STEP}"

while true; do
  if [[ -f "${MONITOR_CSV}" ]]; then
    last_epoch="$(tail -n 1 "${MONITOR_CSV}" | cut -d',' -f1 | tr -d '[:space:]')"
    if [[ "${last_epoch}" =~ ^[0-9]+$ ]]; then
      while [[ "${last_epoch}" -ge "${next_epoch}" ]]; do
        marker_file="${STATE_DIR}/epoch_${next_epoch}.submitted"
        if [[ ! -f "${marker_file}" ]]; then
          eval_output_dir="${OUTPUT_ROOT}/epoch_${next_epoch}"
          mkdir -p "${eval_output_dir}"
          echo "Submitting evaluation for epoch ${next_epoch} using ${CHECKPOINT_NAME}"
          sbatch \
            --job-name="pdac-eval-e${next_epoch}" \
            --partition=gpu \
            --nodes=1 \
            --ntasks=1 \
            --gres=gpu:1 \
            --cpus-per-task=8 \
            --mem=64G \
            --time=08:00:00 \
            --output="${PROJECT_ROOT}/logs/eval_epoch_${next_epoch}_%j.out" \
            --wrap="module load miniconda/3.13 && source \$(conda info --base)/etc/profile.d/conda.sh && conda activate ${CONDA_ENV_NAME} && cd ${PROJECT_ROOT} && radiogenpdac evaluate-encoder-model --pdac-root ${PDAC_ROOT} --nnunet-raw-dir ${NNUNET_RAW_DIR} --nnunet-preprocessed-dir ${NNUNET_PREPROCESSED_DIR} --nnunet-results-dir ${NNUNET_RESULTS_DIR} --model-training-output-dir ${MODEL_TRAINING_OUTPUT_DIR} --images-folder ${IMAGES_FOLDER} --reference-folder ${REFERENCE_FOLDER} --output-folder ${eval_output_dir} --reference-tumor-label ${REFERENCE_TUMOR_LABEL} --prediction-tumor-label ${PREDICTION_TUMOR_LABEL} --split-json ${SPLIT_JSON} --fold ${FOLD} --checkpoint-name ${CHECKPOINT_NAME} --device cuda"
          touch "${marker_file}"
        fi
        next_epoch="$((next_epoch + EPOCH_STEP))"
      done
    fi
  fi

  if [[ -f "${CHECKPOINT_FINAL}" ]]; then
    echo "Training finished; stopping eval submit loop."
    break
  fi
  sleep "${POLL_INTERVAL_SEC}"
done
