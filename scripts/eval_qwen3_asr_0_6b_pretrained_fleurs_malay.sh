#!/bin/bash
#SBATCH --job-name=qwen3_asr_ms_base_eval
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --nodelist=smc-pod-3,smc-pod-4,smc-pod-5,smc-pod-8,smc-pod-9,smc-pod-10
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --chdir=/mnt/weka/aisg/speech_spoke/donghang/project/Qwen3-ASR
#SBATCH --output=/mnt/weka/aisg/speech_spoke/donghang/project/Qwen3-ASR/logs/%x-%j.out
#SBATCH --error=/mnt/weka/aisg/speech_spoke/donghang/project/Qwen3-ASR/logs/%x-%j.err

set -euo pipefail

PROJECT_DIR=${PROJECT_DIR:-/mnt/weka/aisg/speech_spoke/donghang/project/Qwen3-ASR}
ENV_DIR=${ENV_DIR:-/mnt/weka/aisg/speech_spoke/donghang/envs/qwen3asr}
MODEL_PATH=${MODEL_PATH:-/mnt/weka/aisg/speech_spoke/donghang/hf_cache/hub/models--Qwen--Qwen3-ASR-0.6B/snapshots/5eb144179a02acc5e5ba31e748d22b0cf3e303b0}
TEST_JSONL=${TEST_JSONL:-/mnt/weka/aisg/speech_spoke/donghang/data/fleur/malay/test.jsonl}
OUTPUT_DIR=${OUTPUT_DIR:-${PROJECT_DIR}/outputs/fleurs_malay_eval_pretrained_0_6b}

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
nvidia-smi || true

"${PYTHON}" scripts/eval_qwen3_asr_manifest.py \
  --model_path "${MODEL_PATH}" \
  --test_jsonl "${TEST_JSONL}" \
  --output_dir "${OUTPUT_DIR}" \
  --language Malay \
  --batch_size "${BATCH_SIZE:-8}" \
  --max_new_tokens "${MAX_NEW_TOKENS:-256}" \
  --lowercase \
  --remove_punct \
  ${EXTRA_ARGS:-}

echo "End time: $(date)"
