#!/bin/bash
#SBATCH --job-name=qwen3_asr_manifest_eval_alt
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --nodelist=smc-pod-3,smc-pod-4,smc-pod-5,smc-pod-8,smc-pod-9,smc-pod-10
#SBATCH --mem=24G
#SBATCH --time=08:00:00
#SBATCH --chdir=/mnt/weka/aisg/speech_spoke/donghang/project/Qwen3-ASR
#SBATCH --output=/mnt/weka/aisg/speech_spoke/donghang/project/Qwen3-ASR/logs/%x-%j.out
#SBATCH --error=/mnt/weka/aisg/speech_spoke/donghang/project/Qwen3-ASR/logs/%x-%j.err

set -euo pipefail

PROJECT_DIR=${PROJECT_DIR:-/mnt/weka/aisg/speech_spoke/donghang/project/Qwen3-ASR}
ENV_DIR=${ENV_DIR:-/mnt/weka/aisg/speech_spoke/donghang/envs/qwen3-asr}
: "${MODEL_PATH:?MODEL_PATH is required}"
: "${TEST_JSONL:?TEST_JSONL is required}"
: "${OUTPUT_DIR:?OUTPUT_DIR is required}"

source "${ENV_DIR}/bin/activate"
PYTHON=${PYTHON:-$(command -v python)}

export PYTHONUNBUFFERED=1
export HF_HOME=${HF_HOME:-/mnt/weka/aisg/speech_spoke/donghang/hf_cache}
export TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}

mkdir -p "${PROJECT_DIR}/logs" "${OUTPUT_DIR}"
cd "${PROJECT_DIR}"

echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Node: $(hostname)"
echo "Start time: $(date)"
echo "Python: ${PYTHON}"
echo "Model path: ${MODEL_PATH}"
echo "Test JSONL: ${TEST_JSONL}"
echo "Output dir: ${OUTPUT_DIR}"
echo "Language: ${LANGUAGE:-<auto>}"
nvidia-smi || true

CMD=(
  "${PYTHON}" scripts/eval_qwen3_asr_manifest.py
  --model_path "${MODEL_PATH}"
  --test_jsonl "${TEST_JSONL}"
  --output_dir "${OUTPUT_DIR}"
  --batch_size "${BATCH_SIZE:-4}"
  --max_new_tokens "${MAX_NEW_TOKENS:-512}"
)

if [[ -n "${LANGUAGE:-}" ]]; then
  CMD+=(--language "${LANGUAGE}")
fi
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

"${CMD[@]}" ${EXTRA_ARGS:-}

echo "End time: $(date)"
