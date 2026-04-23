#!/bin/bash
# Build a hybrid venous vessel manifest by running baseline inference once,
# extracting multiple vessel labels, and unioning each with the matching manual
# mask. Uses 2 GPUs and resumes from any completed filename-based predictions.
#
# Submit with:
#   sbatch scripts/slurm/venous_full_hybrid_vein_2gpu.sbatch
#
# Optional overrides:
#   PROJECT_ROOT=/path/to/RadiogenPDAC
#   CONDA_ENV_NAME=pdac-ft
#   NNUNET_RAW_DIR=/path/to/nnUNet_raw
#   NNUNET_PREPROCESSED_DIR=/path/to/nnUNet_preprocessed
#   NNUNET_RESULTS_DIR=/path/to/nnUNet_results
#   PANDX_WORKSPACE=/path/to/pandx_workspace
#   DATA_ROOT=/path/to/data
#   PHASE_MANIFEST=/path/to/venous_training_manifest.csv
#   OUTPUT_MANIFEST=/path/to/venous_training_manifest.hybrid_vessels.csv
#   OUTPUT_MASK_DIR=/path/to/hybrid_vessel_masks
#   STRUCTURE_LABELS=vein:2,artery:3
#   NUM_GPUS=2
#   GPU_IDS=0:1
#   REUSE_PREDICTION_DIRS=/path/to/existing/predictions,/path/to/other/predictions
#   REUSE_ONLY=false
#   CHECKPOINT_NAME=checkpoint_final.pth
#   FOLD=0

#SBATCH --job-name=pdac-ven-hybrid-vess
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --output=logs/venous_full_hybrid_vessels_%j.out

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$PWD}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-pdac-ft}"
REPO_PARENT="$(cd "${PROJECT_ROOT}/.." && pwd)"
NNUNET_RAW_DIR="${NNUNET_RAW_DIR:-${REPO_PARENT}/nnUNet_raw}"
NNUNET_PREPROCESSED_DIR="${NNUNET_PREPROCESSED_DIR:-${REPO_PARENT}/nnUNet_preprocessed}"
NNUNET_RESULTS_DIR="${NNUNET_RESULTS_DIR:-${REPO_PARENT}/nnUNet_results}"
PANDX_WORKSPACE="${PANDX_WORKSPACE:-${REPO_PARENT}/pandx_workspace}"
DATA_ROOT="${DATA_ROOT:-${REPO_PARENT}/data}"
PHASE_MANIFEST="${PHASE_MANIFEST:-${DATA_ROOT}/manifests/venous_full/venous_training_manifest.csv}"
OUTPUT_MANIFEST="${OUTPUT_MANIFEST:-${DATA_ROOT}/manifests/venous_full/venous_training_manifest.hybrid_vessels.csv}"
OUTPUT_MASK_DIR="${OUTPUT_MASK_DIR:-${DATA_ROOT}/manifests/venous_full/hybrid_vessel_masks}"
STRUCTURE_LABELS="${STRUCTURE_LABELS:-vein:2,artery:3}"
NUM_GPUS="${NUM_GPUS:-}"
GPU_IDS="${GPU_IDS:-}"
REUSE_PREDICTION_DIRS="${REUSE_PREDICTION_DIRS:-${DATA_ROOT}/manifests/venous_full/fold_${FOLD:-0}_baseline_eval/predictions}"
REUSE_ONLY="${REUSE_ONLY:-false}"
CHECKPOINT_NAME="${CHECKPOINT_NAME:-checkpoint_final.pth}"
FOLD="${FOLD:-0}"
MODEL_TRAINING_OUTPUT_DIR="${MODEL_TRAINING_OUTPUT_DIR:-${PANDX_WORKSPACE}/nnUNet_results/Dataset107_PDAC_Detection/nnUNetTrainerCELossLesionSplit__nnUNetPlans_v3__3d_fullres}"
PDAC_ROOT="${PDAC_ROOT:-${PROJECT_ROOT}/PDAC_Detection}"
PYTHON_BIN="${PYTHON_BIN:-/home/felipe/.conda/envs/${CONDA_ENV_NAME}/bin/python}"

if [[ -n "${NUM_GPUS}" ]]; then
  if ! [[ "${NUM_GPUS}" =~ ^[0-9]+$ ]] || [[ "${NUM_GPUS}" -lt 1 ]]; then
    echo "NUM_GPUS must be a positive integer, got: ${NUM_GPUS}"
    exit 1
  fi
  GPU_IDS=""
  for ((gpu_index=0; gpu_index<NUM_GPUS; gpu_index++)); do
    if [[ -n "${GPU_IDS}" ]]; then
      GPU_IDS+=","
    fi
    GPU_IDS+="${gpu_index}"
  done
elif [[ -z "${GPU_IDS}" ]]; then
  if [[ -n "${SLURM_GPUS_ON_NODE:-}" && "${SLURM_GPUS_ON_NODE}" =~ ^[0-9]+$ ]]; then
    GPU_IDS=""
    for ((gpu_index=0; gpu_index<SLURM_GPUS_ON_NODE; gpu_index++)); do
      if [[ -n "${GPU_IDS}" ]]; then
        GPU_IDS+=","
      fi
      GPU_IDS+="${gpu_index}"
    done
  elif [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    GPU_IDS="${CUDA_VISIBLE_DEVICES}"
  else
    GPU_IDS="0,1"
  fi
fi
GPU_IDS="${GPU_IDS//:/,}"

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
echo "STRUCTURE_LABELS=${STRUCTURE_LABELS}"
echo "NUM_GPUS=${NUM_GPUS}"
echo "GPU_IDS=${GPU_IDS}"
echo "SLURM_GPUS_ON_NODE=${SLURM_GPUS_ON_NODE:-}"
echo "SLURM_JOB_GPUS=${SLURM_JOB_GPUS:-}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
echo "REUSE_PREDICTION_DIRS=${REUSE_PREDICTION_DIRS}"
echo "REUSE_ONLY=${REUSE_ONLY}"

"${PYTHON_BIN}" -V
"${PYTHON_BIN}" -c "import acvl_utils; print('acvl_utils ok:', acvl_utils.__file__)"

STRUCTURE_LABEL_ARGS=()
IFS=',' read -r -a STRUCTURE_LABEL_ITEMS <<< "${STRUCTURE_LABELS}"
for item in "${STRUCTURE_LABEL_ITEMS[@]}"; do
  STRUCTURE_LABEL_ARGS+=(--structure-label "${item}")
done

GPU_ID_ARGS=()
IFS=',' read -r -a GPU_ID_ITEMS <<< "${GPU_IDS}"
for item in "${GPU_ID_ITEMS[@]}"; do
  if [[ -n "${item}" ]]; then
    GPU_ID_ARGS+=(--gpu-id "${item}")
  fi
done
echo "PARSED_GPU_IDS=${GPU_ID_ARGS[*]}"

REUSE_PREDICTION_ARGS=()
IFS=',' read -r -a REUSE_PREDICTION_ITEMS <<< "${REUSE_PREDICTION_DIRS}"
for item in "${REUSE_PREDICTION_ITEMS[@]}"; do
  if [[ -d "${item}" ]]; then
    REUSE_PREDICTION_ARGS+=(--reuse-prediction-dir "${item}")
  else
    echo "Skipping missing reusable prediction dir: ${item}"
  fi
done

REUSE_ONLY_ARGS=()
case "${REUSE_ONLY}" in
  true|True|TRUE|1|yes|Yes|YES)
    REUSE_ONLY_ARGS+=(--reuse-only)
    ;;
esac

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
  "${STRUCTURE_LABEL_ARGS[@]}" \
  "${REUSE_PREDICTION_ARGS[@]}" \
  "${REUSE_ONLY_ARGS[@]}" \
  --checkpoint-name "${CHECKPOINT_NAME}" \
  --phase venous \
  --fold "${FOLD}" \
  "${GPU_ID_ARGS[@]}"
