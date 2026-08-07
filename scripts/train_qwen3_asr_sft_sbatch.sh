#!/bin/bash
#SBATCH --job-name=qwen3_asr_sft
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --nodelist=smc-pod-3,smc-pod-4,smc-pod-5,smc-pod-8,smc-pod-9,smc-pod-10
#SBATCH --mem=48G
#SBATCH --time=48:00:00
#SBATCH --chdir=/mnt/weka/aisg/speech_spoke/donghang/project/Qwen3-ASR
#SBATCH --output=/mnt/weka/aisg/speech_spoke/donghang/project/Qwen3-ASR/logs/%x-%j.out
#SBATCH --error=/mnt/weka/aisg/speech_spoke/donghang/project/Qwen3-ASR/logs/%x-%j.err

set -euo pipefail

PROJECT_DIR=${PROJECT_DIR:-/mnt/weka/aisg/speech_spoke/donghang/project/Qwen3-ASR}
ENV_DIR=${ENV_DIR:-/mnt/weka/aisg/speech_spoke/donghang/envs/qwen3asr}
: "${MODEL_PATH:?MODEL_PATH is required}"
: "${TRAIN_FILE:?TRAIN_FILE is required}"
: "${OUTPUT_DIR:?OUTPUT_DIR is required}"

source "${ENV_DIR}/bin/activate"
PYTHON=${PYTHON:-$(command -v python)}

export PYTHONUNBUFFERED=1
export HF_HOME=${HF_HOME:-/mnt/weka/aisg/speech_spoke/donghang/hf_cache}
export TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}
export HF_DATASETS_OFFLINE=${HF_DATASETS_OFFLINE:-1}

mkdir -p "${PROJECT_DIR}/logs" "${OUTPUT_DIR}"
cd "${PROJECT_DIR}"

echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Node: $(hostname)"
echo "Start time: $(date)"
echo "Python: ${PYTHON}"
echo "Init model path: ${MODEL_PATH}"
echo "Train file: ${TRAIN_FILE}"
echo "Eval file: ${EVAL_FILE:-}"
echo "Output dir: ${OUTPUT_DIR}"
echo "Config: batch_size=${BATCH_SIZE:-1}, grad_acc=${GRAD_ACC:-16}, lr=${LR:-5e-6}, epochs=${EPOCHS:-1}"
nvidia-smi || true

CMD=(
  "${PYTHON}" finetuning/qwen3_asr_sft.py
  --model_path "${MODEL_PATH}"
  --train_file "${TRAIN_FILE}"
  --output_dir "${OUTPUT_DIR}"
  --batch_size "${BATCH_SIZE:-1}"
  --grad_acc "${GRAD_ACC:-16}"
  --lr "${LR:-5e-6}"
  --epochs "${EPOCHS:-1}"
  --max_steps "${MAX_STEPS:--1}"
  --log_steps "${LOG_STEPS:-20}"
  --save_strategy "${SAVE_STRATEGY:-steps}"
  --save_steps "${SAVE_STEPS:-1000}"
  --save_total_limit "${SAVE_TOTAL_LIMIT:-5}"
  --num_workers "${NUM_WORKERS:-2}"
  --pin_memory "${PIN_MEMORY:-1}"
  --persistent_workers "${PERSISTENT_WORKERS:-1}"
  --prefetch_factor "${PREFETCH_FACTOR:-2}"
)

if [[ -n "${EVAL_FILE:-}" ]]; then
  CMD+=(--eval_file "${EVAL_FILE}")
fi

if [[ "${FORCE_ASR_LM_HEAD:-0}" == "1" ]]; then
  CMD+=(--force_asr_lm_head)
fi

"${CMD[@]}" ${EXTRA_ARGS:-}

echo "End time: $(date)"
