#!/bin/bash
# Refresh the full venous training manifests from the current cluster data tree.
# This is intended to be run before hybrid vessel-mask generation when new venous
# cases have been added under data/volumes/venous.
#
# Important:
# - Discovery already ignores macOS metadata files like ._* and .DS_Store.
# - We only require tumor + pancreas masks here so artery/vein can be supplied
#   later by the baseline model during the hybrid-vessel build.
#
# Submit with:
#   sbatch scripts/slurm/venous_full_refresh_manifests.sbatch
#
# Optional overrides:
#   PROJECT_ROOT=/path/to/RadiogenPDAC
#   CONDA_ENV_NAME=pdac-ft
#   DATA_ROOT=/path/to/data
#   OUTPUT_DIR=/path/to/data/manifests/venous_full
#   REQUIRED_STRUCTURES=tumor,pancreas
#   PYTHON_BIN=/abs/path/to/python

#SBATCH --job-name=pdac-ven-refresh-mf
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=logs/venous_full_refresh_manifests_%j.out

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$PWD}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-pdac-ft}"
DATA_ROOT="${DATA_ROOT:-$(cd "${PROJECT_ROOT}/.." && pwd)/data}"
OUTPUT_DIR="${OUTPUT_DIR:-${DATA_ROOT}/manifests/venous_full}"
REQUIRED_STRUCTURES="${REQUIRED_STRUCTURES:-tumor,pancreas}"
PYTHON_BIN="${PYTHON_BIN:-/home/felipe/.conda/envs/${CONDA_ENV_NAME}/bin/python}"

mkdir -p "${PROJECT_ROOT}/logs" "${OUTPUT_DIR}"

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
echo "DATA_ROOT=${DATA_ROOT}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"
echo "REQUIRED_STRUCTURES=${REQUIRED_STRUCTURES}"

"${PYTHON_BIN}" -V

echo "[scan] refreshing venous manifests"
"${PYTHON_BIN}" -m radiogenpdac.cli \
  scan-cluster-complete-cases \
  --framework-root "${PROJECT_ROOT}" \
  --data-root "${DATA_ROOT}" \
  --output-dir "${OUTPUT_DIR}" \
  --phases venous \
  --required-structures "${REQUIRED_STRUCTURES}"

VENOUS_MANIFEST="${OUTPUT_DIR}/venous_training_manifest.csv"
DISCOVERED_MANIFEST="${OUTPUT_DIR}/cluster_phase_manifest.csv"
INVENTORY_MANIFEST="${OUTPUT_DIR}/cluster_case_inventory.csv"

echo "[counts] discovered venous rows: $(awk -F',' 'NR>1 && $2==\"venous\" {count++} END {print count+0}' "${DISCOVERED_MANIFEST}")"
echo "[counts] complete venous rows: $(tail -n +2 "${VENOUS_MANIFEST}" | wc -l | tr -d ' ')"
echo "[counts] inventory rows: $(tail -n +2 "${INVENTORY_MANIFEST}" | wc -l | tr -d ' ')"
echo "[done] venous manifest: ${VENOUS_MANIFEST}"
