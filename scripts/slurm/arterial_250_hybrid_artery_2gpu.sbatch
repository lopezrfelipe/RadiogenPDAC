#!/bin/bash
# Build a hybrid arterial artery manifest by unioning manual artery masks with
# baseline-model artery predictions. Uses 2 GPUs and resumes from any completed
# filename-based predictions already present in the output directory.
#
# Submit with:
#   sbatch scripts/slurm/arterial_250_hybrid_artery_2gpu.sbatch
#
# Optional overrides:
#   PROJECT_ROOT=/path/to/RadiogenPDAC
#   CONDA_ENV_NAME=pdac-ft
#   NNUNET_RAW_DIR=/path/to/nnUNet_raw
#   NNUNET_PREPROCESSED_DIR=/path/to/nnUNet_preprocessed
#   NNUNET_RESULTS_DIR=/path/to/nnUNet_results
#   PANDX_WORKSPACE=/path/to/pandx_workspace
#   DATA_ROOT=/path/to/data
#   PHASE_MANIFEST=/path/to/arterial_training_manifest.csv
#   OUTPUT_MANIFEST=/path/to/arterial_training_manifest.hybrid.csv
#   OUTPUT_MASK_DIR=/path/to/hybrid_artery_masks
#   STRUCTURE_NAME=artery
#   PREDICTION_LABEL=3
#   CHECKPOINT_NAME=checkpoint_final.pth
#   FOLD=0

#SBATCH --job-name=pdac-art-hybrid-art
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --output=logs/arterial_250_hybrid_artery_%j.out

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$PWD}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-pdac-ft}"
REPO_PARENT="$(cd "${PROJECT_ROOT}/.." && pwd)"
NNUNET_RAW_DIR="${NNUNET_RAW_DIR:-${REPO_PARENT}/nnUNet_raw}"
NNUNET_PREPROCESSED_DIR="${NNUNET_PREPROCESSED_DIR:-${REPO_PARENT}/nnUNet_preprocessed}"
NNUNET_RESULTS_DIR="${NNUNET_RESULTS_DIR:-${REPO_PARENT}/nnUNet_results}"
PANDX_WORKSPACE="${PANDX_WORKSPACE:-${REPO_PARENT}/pandx_workspace}"
DATA_ROOT="${DATA_ROOT:-${REPO_PARENT}/data}"
PHASE_MANIFEST="${PHASE_MANIFEST:-${DATA_ROOT}/manifests/complete_cases/arterial_training_manifest.csv}"
OUTPUT_MANIFEST="${OUTPUT_MANIFEST:-${DATA_ROOT}/manifests/arterial_250/arterial_training_manifest.hybrid.csv}"
OUTPUT_MASK_DIR="${OUTPUT_MASK_DIR:-${DATA_ROOT}/manifests/arterial_250/hybrid_artery_masks}"
STRUCTURE_NAME="${STRUCTURE_NAME:-artery}"
PREDICTION_LABEL="${PREDICTION_LABEL:-3}"
CHECKPOINT_NAME="${CHECKPOINT_NAME:-checkpoint_final.pth}"
FOLD="${FOLD:-0}"
MODEL_TRAINING_OUTPUT_DIR="${MODEL_TRAINING_OUTPUT_DIR:-${PANDX_WORKSPACE}/nnUNet_results/Dataset107_PDAC_Detection/nnUNetTrainerCELossLesionSplit__nnUNetPlans_v3__3d_fullres}"
PDAC_ROOT="${PDAC_ROOT:-${PROJECT_ROOT}/PDAC_Detection}"
PYTHON_BIN="${PYTHON_BIN:-/home/felipe/.conda/envs/${CONDA_ENV_NAME}/bin/python}"

mkdir -p "${PROJECT_ROOT}/logs"

if [[ ! -f "${PHASE_MANIFEST}" ]]; then
  echo "Missing phase manifest: ${PHASE_MANIFEST}"
  exit 1
fi

if [[ ! -d "${MODEL_TRAINING_OUTPUT_DIR}" ]]; then
  echo "Missing model training output dir: ${MODEL_TRAINING_OUTPUT_DIR}"
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

echo "HOST=$(hostname)"
echo "PYTHON_BIN=${PYTHON_BIN}"
echo "PHASE_MANIFEST=${PHASE_MANIFEST}"
echo "OUTPUT_MANIFEST=${OUTPUT_MANIFEST}"
echo "OUTPUT_MASK_DIR=${OUTPUT_MASK_DIR}"

"${PYTHON_BIN}" -V
"${PYTHON_BIN}" -c "import acvl_utils; print('acvl_utils ok:', acvl_utils.__file__)"

"${PYTHON_BIN}" -m radiogenpdac.cli \
  build-hybrid-structure-manifest-from-model \
  --phase-manifest "${PHASE_MANIFEST}" \
  --output-manifest "${OUTPUT_MANIFEST}" \
  --output-mask-dir "${OUTPUT_MASK_DIR}" \
  --pdac-root "${PDAC_ROOT}" \
  --nnunet-raw-dir "${NNUNET_RAW_DIR}" \
  --nnunet-preprocessed-dir "${NNUNET_PREPROCESSED_DIR}" \
  --nnunet-results-dir "${NNUNET_RESULTS_DIR}" \
  --model-training-output-dir "${MODEL_TRAINING_OUTPUT_DIR}" \
  --structure-name "${STRUCTURE_NAME}" \
  --prediction-label "${PREDICTION_LABEL}" \
  --checkpoint-name "${CHECKPOINT_NAME}" \
  --phase arterial \
  --fold "${FOLD}" \
  --gpu-id 0 \
  --gpu-id 1
