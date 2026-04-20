#!/bin/bash
# Smoke-test venous multiclass fine-tuning on up to 50 complete cases.
# Submit with:
#   sbatch --array=0-1 scripts/slurm/venous_smoke50_array.sbatch
#
# Required environment variables:
#   PRETRAINED_WEIGHTS=/abs/path/to/checkpoint_final.pth
#   ORIGINAL_MODEL_TRAINING_OUTPUT_DIR=/abs/path/to/original/model_training_output_dir
#
# Optional overrides:
#   PROJECT_ROOT=/path/to/RadiogenPDAC
#   CONDA_ENV_NAME=pdac-ft
#   PHASE=venous
#   SMOKE_CASES=50
#   DATA_ROOT=/path/to/data
#   NNUNET_RAW_DIR=/path/to/nnUNet_raw
#   NNUNET_PREPROCESSED_DIR=/path/to/nnUNet_preprocessed
#   NNUNET_RESULTS_DIR=/path/to/nnUNet_results
#   DATASET_ID=901
#   DATASET_NAME=PDACVenousSmoke50
#   WORKFLOW_ROOT=/path/to/data/manifests/workflows/venous_smoke_50
#   GPU_MEMORY_TARGET_GB=40
#   NUM_PROCESSES=8
#   TRAINER_NAME=nnUNetTrainer_ftce
#   PLANS_IDENTIFIER=nnUNetPlans
#   CONFIGURATION=3d_fullres

#SBATCH --job-name=pdac-ven-smoke50
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --output=logs/venous_smoke50_%A_%a.out

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$PWD}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-pdac-ft}"
PHASE="${PHASE:-venous}"
SMOKE_CASES="${SMOKE_CASES:-50}"
DATA_ROOT="${DATA_ROOT:-$(cd "${PROJECT_ROOT}/.." && pwd)/data}"
NNUNET_RAW_DIR="${NNUNET_RAW_DIR:-$(cd "${PROJECT_ROOT}/.." && pwd)/nnUNet_raw}"
NNUNET_PREPROCESSED_DIR="${NNUNET_PREPROCESSED_DIR:-$(cd "${PROJECT_ROOT}/.." && pwd)/nnUNet_preprocessed}"
NNUNET_RESULTS_DIR="${NNUNET_RESULTS_DIR:-$(cd "${PROJECT_ROOT}/.." && pwd)/nnUNet_results}"
DATASET_ID="${DATASET_ID:-901}"
DATASET_NAME="${DATASET_NAME:-PDACVenousSmoke50}"
WORKFLOW_ROOT="${WORKFLOW_ROOT:-${DATA_ROOT}/manifests/workflows/venous_smoke_50}"
SMOKE_MANIFEST_DIR="${DATA_ROOT}/manifests/smoke"
SCAN_OUTPUT_DIR="${SMOKE_MANIFEST_DIR}"
FULL_MANIFEST="${SCAN_OUTPUT_DIR}/${PHASE}_training_manifest.csv"
SUBSET_MANIFEST="${SCAN_OUTPUT_DIR}/${PHASE}_training_manifest.smoke${SMOKE_CASES}.csv"
PREPARED_INDEX="${SCAN_OUTPUT_DIR}/${PHASE,,}_smoke${SMOKE_CASES}_prepared_index.csv"
SPLITS_JSON="${NNUNET_PREPROCESSED_DIR}/Dataset$(printf "%03d" "${DATASET_ID}")_${DATASET_NAME}/splits_final.json"
GPU_MEMORY_TARGET_GB="${GPU_MEMORY_TARGET_GB:-40}"
NUM_PROCESSES="${NUM_PROCESSES:-8}"
TRAINER_NAME="${TRAINER_NAME:-nnUNetTrainer_ftce}"
PLANS_IDENTIFIER="${PLANS_IDENTIFIER:-nnUNetPlans}"
CONFIGURATION="${CONFIGURATION:-3d_fullres}"
FOLD="${SLURM_ARRAY_TASK_ID:-0}"
LOCK_DIR="${WORKFLOW_ROOT}/.prep.lock"
DONE_FILE="${WORKFLOW_ROOT}/.prep.done"

if [[ -z "${PRETRAINED_WEIGHTS:-}" ]]; then
  echo "PRETRAINED_WEIGHTS is required"
  exit 1
fi

if [[ -z "${ORIGINAL_MODEL_TRAINING_OUTPUT_DIR:-}" ]]; then
  echo "ORIGINAL_MODEL_TRAINING_OUTPUT_DIR is required"
  exit 1
fi

mkdir -p "${PROJECT_ROOT}/logs" "${SMOKE_MANIFEST_DIR}" "${WORKFLOW_ROOT}" "${NNUNET_RAW_DIR}" "${NNUNET_PREPROCESSED_DIR}" "${NNUNET_RESULTS_DIR}"

cd "${PROJECT_ROOT}"

module load miniconda
eval "$(conda shell.bash hook)"
conda activate "${CONDA_ENV_NAME}"

run_shared_prep() {
  echo "[prep] scanning complete ${PHASE} cases"
  radiogenpdac scan-cluster-complete-cases \
    --framework-root . \
    --phases "${PHASE}" \
    --required-structures tumor,pancreas,artery,vein \
    --output-dir "${SCAN_OUTPUT_DIR}"

  echo "[prep] building ${SMOKE_CASES}-case subset manifest"
  awk -F',' -v limit="$((SMOKE_CASES + 1))" 'NR==1 || NR<=limit' "${FULL_MANIFEST}" > "${SUBSET_MANIFEST}"

  case_count="$(tail -n +2 "${SUBSET_MANIFEST}" | wc -l | tr -d ' ')"
  echo "[prep] subset contains ${case_count} cases"
  if [[ "${case_count}" -lt 2 ]]; then
    echo "Need at least 2 cases in ${SUBSET_MANIFEST}, found ${case_count}"
    exit 1
  fi

  echo "[prep] preparing nnU-Net dataset"
  radiogenpdac prepare-ingested-encoder-dataset \
    --phase-manifest "${SUBSET_MANIFEST}" \
    --phase "${PHASE}" \
    --dataset-id "${DATASET_ID}" \
    --dataset-name "${DATASET_NAME}" \
    --pdac-root "${PROJECT_ROOT}/PDAC_Detection" \
    --nnunet-raw-dir "${NNUNET_RAW_DIR}" \
    --output-index "${PREPARED_INDEX}" \
    --task-mode multiclass \
    --crop-mode pancreas_roi

  echo "[prep] writing splits"
  radiogenpdac write-encoder-splits \
    --prepared-index "${PREPARED_INDEX}" \
    --nnunet-preprocessed-dir "${NNUNET_PREPROCESSED_DIR}" \
    --dataset-id "${DATASET_ID}" \
    --dataset-name "${DATASET_NAME}" \
    --output-json "${SPLITS_JSON}" \
    --n-folds 2 \
    --seed 12345

  echo "[prep] planning/preprocessing nnU-Net dataset"
  radiogenpdac plan-encoder-dataset \
    --dataset-id "${DATASET_ID}" \
    --pdac-root "${PROJECT_ROOT}/PDAC_Detection" \
    --nnunet-raw-dir "${NNUNET_RAW_DIR}" \
    --nnunet-preprocessed-dir "${NNUNET_PREPROCESSED_DIR}" \
    --nnunet-results-dir "${NNUNET_RESULTS_DIR}" \
    --plans-identifier "${PLANS_IDENTIFIER}" \
    --configurations "${CONFIGURATION}" \
    --gpu-memory-target-gb "${GPU_MEMORY_TARGET_GB}" \
    --num-processes "${NUM_PROCESSES}"

  touch "${DONE_FILE}"
}

if [[ ! -f "${DONE_FILE}" ]]; then
  if mkdir "${LOCK_DIR}" 2>/dev/null; then
    trap 'rmdir "${LOCK_DIR}"' EXIT
    run_shared_prep
    rmdir "${LOCK_DIR}"
    trap - EXIT
  else
    echo "[prep] waiting for shared preparation to finish"
    while [[ ! -f "${DONE_FILE}" ]]; do
      sleep 20
    done
  fi
fi

DATASET_DIR="${NNUNET_RAW_DIR}/Dataset$(printf "%03d" "${DATASET_ID}")_${DATASET_NAME}"
MODEL_DIR="${NNUNET_RESULTS_DIR}/Dataset$(printf "%03d" "${DATASET_ID}")_${DATASET_NAME}/${TRAINER_NAME}__${PLANS_IDENTIFIER}__${CONFIGURATION}"
FOLD_WORKFLOW_DIR="${WORKFLOW_ROOT}/fold_${FOLD}"
BASELINE_DIR="${FOLD_WORKFLOW_DIR}/baseline_eval"
POST_JSON="${FOLD_WORKFLOW_DIR}/post_finetune_validation_tumor_metrics.json"
MONITOR_JSON="${FOLD_WORKFLOW_DIR}/training_monitor.fold_${FOLD}.json"
MONITOR_CSV="${FOLD_WORKFLOW_DIR}/training_monitor.fold_${FOLD}.csv"

mkdir -p "${FOLD_WORKFLOW_DIR}" "${BASELINE_DIR}"

echo "[fold ${FOLD}] baseline evaluation"
radiogenpdac evaluate-encoder-model \
  --pdac-root "${PROJECT_ROOT}/PDAC_Detection" \
  --nnunet-raw-dir "${NNUNET_RAW_DIR}" \
  --nnunet-preprocessed-dir "${NNUNET_PREPROCESSED_DIR}" \
  --nnunet-results-dir "${NNUNET_RESULTS_DIR}" \
  --model-training-output-dir "${ORIGINAL_MODEL_TRAINING_OUTPUT_DIR}" \
  --images-folder "${DATASET_DIR}/imagesTr" \
  --reference-folder "${DATASET_DIR}/labelsTr" \
  --output-folder "${BASELINE_DIR}" \
  --reference-tumor-label 2 \
  --prediction-tumor-label 1 \
  --split-json "${SPLITS_JSON}" \
  --fold "${FOLD}" \
  --checkpoint-name checkpoint_final.pth \
  --device cuda

echo "[fold ${FOLD}] fine-tuning"
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
  --num-gpus 2

echo "[fold ${FOLD}] writing training monitor snapshot"
radiogenpdac monitor-encoder-training \
  --model-training-output-dir "${MODEL_DIR}" \
  --fold "${FOLD}" \
  --output-json "${MONITOR_JSON}" \
  --output-csv "${MONITOR_CSV}"

echo "[fold ${FOLD}] post-fine-tune validation tumor metrics"
radiogenpdac summarize-validation-tumor-metrics \
  --reference-folder "${NNUNET_PREPROCESSED_DIR}/Dataset$(printf "%03d" "${DATASET_ID}")_${DATASET_NAME}/gt_segmentations" \
  --prediction-folder "${MODEL_DIR}/fold_${FOLD}/validation" \
  --output-json "${POST_JSON}" \
  --reference-tumor-label 2 \
  --prediction-tumor-label 2

echo "[fold ${FOLD}] done"
echo "Baseline outputs: ${BASELINE_DIR}"
echo "Post-finetune tumor metrics: ${POST_JSON}"
echo "Training monitor: ${MONITOR_JSON}"
