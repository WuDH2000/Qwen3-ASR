#!/bin/bash
#SBATCH --job-name=qwen3_asr_ms_sft
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --nodelist=smc-pod-3,smc-pod-4,smc-pod-5,smc-pod-8,smc-pod-9,smc-pod-10
#SBATCH --mem=80G
#SBATCH --time=48:00:00
#SBATCH --chdir=/mnt/weka/aisg/speech_spoke/donghang/project/Qwen3-ASR
#SBATCH --output=/mnt/weka/aisg/speech_spoke/donghang/project/Qwen3-ASR/logs/%x-%j.out
#SBATCH --error=/mnt/weka/aisg/speech_spoke/donghang/project/Qwen3-ASR/logs/%x-%j.err

set -euo pipefail

PROJECT_DIR=${PROJECT_DIR:-/mnt/weka/aisg/speech_spoke/donghang/project/Qwen3-ASR}
ENV_DIR=${ENV_DIR:-/mnt/weka/aisg/speech_spoke/donghang/envs/qwen3asr}
MODEL_PATH=${MODEL_PATH:-/mnt/weka/aisg/speech_spoke/donghang/hf_cache/hub/models--Qwen--Qwen3-ASR-0.6B/snapshots/5eb144179a02acc5e5ba31e748d22b0cf3e303b0}
TRAIN_FILE=${TRAIN_FILE:-/mnt/weka/aisg/speech_spoke/donghang/data/fleur/malay/train.jsonl}
EVAL_FILE=${EVAL_FILE:-/mnt/weka/aisg/speech_spoke/donghang/data/fleur/malay/validation.jsonl}
OUTPUT_DIR=${OUTPUT_DIR:-${PROJECT_DIR}/outputs/qwen3-asr-0.6b-malay-sft}

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
echo "Model path: ${MODEL_PATH}"
echo "Train file: ${TRAIN_FILE}"
echo "Eval file: ${EVAL_FILE}"
echo "Output dir: ${OUTPUT_DIR}"
nvidia-smi || true

"${PYTHON}" finetuning/qwen3_asr_sft.py \
  --model_path "${MODEL_PATH}" \
  --train_file "${TRAIN_FILE}" \
  --eval_file "${EVAL_FILE}" \
  --output_dir "${OUTPUT_DIR}" \
  --batch_size "${BATCH_SIZE:-4}" \
  --grad_acc "${GRAD_ACC:-8}" \
  --lr "${LR:-2e-5}" \
  --epochs "${EPOCHS:-1}" \
  --log_steps "${LOG_STEPS:-10}" \
  --save_strategy "${SAVE_STRATEGY:-steps}" \
  --save_steps "${SAVE_STEPS:-50}" \
  --save_total_limit "${SAVE_TOTAL_LIMIT:-5}" \
  --num_workers "${NUM_WORKERS:-2}" \
  --pin_memory "${PIN_MEMORY:-1}" \
  --persistent_workers "${PERSISTENT_WORKERS:-1}" \
  --prefetch_factor "${PREFETCH_FACTOR:-2}" \
  ${EXTRA_ARGS:-}

echo "End time: $(date)"
