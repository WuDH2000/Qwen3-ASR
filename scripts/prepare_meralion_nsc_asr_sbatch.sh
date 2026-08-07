#!/bin/bash
#SBATCH --job-name=prep_meralion_nsc
#SBATCH --array=1-4
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --nodelist=smc-pod-3,smc-pod-4,smc-pod-5,smc-pod-8,smc-pod-9,smc-pod-10
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --chdir=/mnt/weka/aisg/speech_spoke/donghang/project/Qwen3-ASR
#SBATCH --output=/mnt/weka/aisg/speech_spoke/donghang/project/Qwen3-ASR/logs/%x-%A_%a.out
#SBATCH --error=/mnt/weka/aisg/speech_spoke/donghang/project/Qwen3-ASR/logs/%x-%A_%a.err

set -euo pipefail

PROJECT_DIR=${PROJECT_DIR:-/mnt/weka/aisg/speech_spoke/donghang/project/Qwen3-ASR}
ENV_DIR=${ENV_DIR:-/mnt/weka/aisg/speech_spoke/donghang/envs/qwen3asr}
SOURCE_DIR=${SOURCE_DIR:-/mnt/weka/aisg/speech_spoke/donghang/hf_cache/hub/datasets--MERaLiON--Multitask-National-Speech-Corpus-v1/snapshots/d169debbce4a2dc07f7293360d3eed33c06793b9}
OUTPUT_DIR=${OUTPUT_DIR:-/mnt/weka/aisg/speech_spoke/donghang/data/meralion-multitask-national-speech-corpus-v1}
PART=${PART:-${SLURM_ARRAY_TASK_ID}}

source "${ENV_DIR}/bin/activate"
PYTHON=${PYTHON:-$(command -v python)}

export PYTHONUNBUFFERED=1

mkdir -p "${PROJECT_DIR}/logs" "${OUTPUT_DIR}"
cd "${PROJECT_DIR}"

echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Array task: ${SLURM_ARRAY_TASK_ID:-none}"
echo "Part: ${PART}"
echo "Node: $(hostname)"
echo "Start time: $(date)"
echo "Python: ${PYTHON}"
echo "Source dir: ${SOURCE_DIR}"
echo "Output dir: ${OUTPUT_DIR}"

"${PYTHON}" scripts/prepare_meralion_nsc_asr.py \
  --source-dir "${SOURCE_DIR}" \
  --output-dir "${OUTPUT_DIR}" \
  --parts "${PART}" \
  --batch-size "${BATCH_SIZE:-256}" \
  --skip-metadata

echo "End time: $(date)"
