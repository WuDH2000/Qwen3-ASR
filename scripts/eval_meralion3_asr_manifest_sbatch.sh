#!/bin/bash
#SBATCH --job-name=meralion3_asr_eval
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --nodelist=smc-pod-3,smc-pod-4,smc-pod-5,smc-pod-8,smc-pod-9,smc-pod-10
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --chdir=/mnt/weka/aisg/speech_spoke/donghang/project/Qwen3-ASR
#SBATCH --output=/mnt/weka/aisg/speech_spoke/donghang/project/Qwen3-ASR/logs/%x-%j.out
#SBATCH --error=/mnt/weka/aisg/speech_spoke/donghang/project/Qwen3-ASR/logs/%x-%j.err

set -euo pipefail

PROJECT_DIR=${PROJECT_DIR:-/mnt/weka/aisg/speech_spoke/donghang/project/Qwen3-ASR}
ENV_DIR=${ENV_DIR:-/mnt/weka/aisg/speech_spoke/donghang/envs/qwen3asr}
: "${MODEL_PATH:?MODEL_PATH is required}"
: "${TEST_JSONL:?TEST_JSONL is required}"
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
echo "Model path: ${MODEL_PATH}"
echo "Test JSONL: ${TEST_JSONL}"
echo "Output dir: ${OUTPUT_DIR}"
echo "Backend: ${BACKEND:-transformers}"
nvidia-smi || true

CMD=(
  "${PYTHON}" scripts/eval_meralion3_asr_manifest.py
  --model_path "${MODEL_PATH}"
  --test_jsonl "${TEST_JSONL}"
  --output_dir "${OUTPUT_DIR}"
  --backend "${BACKEND:-transformers}"
  --batch_size "${BATCH_SIZE:-4}"
  --limit "${LIMIT:-0}"
)

if [[ "${LOWERCASE:-1}" == "1" ]]; then
  CMD+=(--lowercase)
fi
if [[ "${REMOVE_PUNCT:-0}" == "1" ]]; then
  CMD+=(--remove_punct)
fi
if [[ "${REMOVE_UNICODE_PUNCT:-1}" == "1" ]]; then
  CMD+=(--remove_unicode_punct)
fi
if [[ "${CJK_CHAR_SPACE:-0}" == "1" ]]; then
  CMD+=(--cjk_char_space)
fi
if [[ "${REMOVE_SPEAKER_TAGS:-0}" == "1" ]]; then
  CMD+=(--remove_speaker_tags)
fi
if [[ "${DRY_RUN:-0}" == "1" ]]; then
  CMD+=(--dry_run)
fi

"${CMD[@]}" ${EXTRA_ARGS:-}

echo "End time: $(date)"
