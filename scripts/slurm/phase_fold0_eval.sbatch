#!/bin/bash
# Backward-compatible alias for the more general phase evaluation job.
exec /bin/bash "$(dirname "$0")/phase_eval.sbatch" "$@"
