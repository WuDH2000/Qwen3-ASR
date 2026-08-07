# Repository Guidelines

## Project Structure & Module Organization

This repository packages Qwen3-ASR inference and finetuning utilities as `qwen-asr`. Core package code lives under `qwen_asr/`: `inference/` contains the public ASR and forced-aligner wrappers, `core/transformers_backend/` and `core/vllm_backend/` contain backend integrations, and `cli/` provides console entrypoints. Runnable usage examples are in `examples/`. Fine-tuning code and data-format notes are in `finetuning/`. Static assets are in `assets/`, Docker build material is in `docker/`, and GitHub automation is under `.github/`.

## Build, Test, and Development Commands

Use a fresh Python 3.9+ environment; the README recommends Python 3.12.

```bash
pip install -e .
pip install -e ".[vllm]"
python -m qwen_asr --help
qwen-asr-demo --asr-checkpoint Qwen/Qwen3-ASR-1.7B
qwen-asr-demo-streaming --asr-checkpoint Qwen/Qwen3-ASR-1.7B
qwen-asr-serve --help
```

`pip install -e .` installs the transformers backend for local development. The `.[vllm]` extra enables the vLLM backend. The console scripts are declared in `pyproject.toml` and should remain the preferred way to exercise CLI behavior. Fine-tuning examples are run from `finetuning/`, for example `torchrun --nproc_per_node=2 qwen3_asr_sft.py --train_file ./train.jsonl --output_dir ./qwen3-asr-finetuning-out`.

## Local Fine-Tuning Context

For the current workspace, use the Qwen3-ASR environment at `/mnt/weka/aisg/speech_spoke/donghang/envs/qwen3asr`:

```bash
source /mnt/weka/aisg/speech_spoke/donghang/envs/qwen3asr/bin/activate
uv pip install <package>
```

The user recreated this environment after an earlier dependency incident. Do not modify other environments. Avoid installing or changing `torch` unless the user explicitly requests it; if a package is needed, activate this env and use `uv pip install <package>`. Direct local GPU use is not available. Submit GPU work through Slurm and include `#SBATCH --nodelist=smc-pod-3,smc-pod-4,smc-pod-5,smc-pod-8,smc-pod-9,smc-pod-10`. Do not switch to other nodes without explicit user approval.

Local model snapshots used in current experiments:

- Qwen3-ASR-0.6B: `/mnt/weka/aisg/speech_spoke/donghang/hf_cache/hub/models--Qwen--Qwen3-ASR-0.6B/snapshots/5eb144179a02acc5e5ba31e748d22b0cf3e303b0`
- Qwen3-ASR-1.7B: `/mnt/weka/aisg/speech_spoke/donghang/hf_cache/hub/models--Qwen--Qwen3-ASR-1.7B/snapshots/7278e1e70fe206f11671096ffdd38061171dd6e5`
- Qwen3-ForcedAligner-0.6B: `/mnt/weka/aisg/speech_spoke/donghang/hf_cache/hub/models--Qwen--Qwen3-ForcedAligner-0.6B`
- Polyglot-Lion-1.7B: `/mnt/weka/aisg/speech_spoke/donghang/hf_cache/hub/models--knoveleng--polyglot-lion-1.7b/snapshots/9ef388b498182f50a5a7909e6e7a4e2d89a840c4`
- MERaLiON-3-3B-ASR: `/mnt/weka/aisg/speech_spoke/donghang/hf_cache/hub/models--MERaLiON--MERaLiON-3-3B-ASR/snapshots/9f226d738dee62c60a2f43b0ab7b9a2585a132c3`

Current data preparation and evaluation scripts live under `scripts/`:

- Tamil: `prepare_deepdml_tamil_asr.py`, `train_qwen3_asr_0_6b_tamil_sft.sh`, `eval_qwen3_asr_fleurs_tamil.py`, `eval_qwen3_asr_fleurs_tamil.sh`, and `eval_qwen3_asr_0_6b_pretrained_fleurs_tamil.sh`.
- Malay: `prepare_fleurs_malay_asr.py`, `train_qwen3_asr_0_6b_malay_sft.sh`, `eval_qwen3_asr_manifest.py`, `eval_qwen3_asr_0_6b_pretrained_fleurs_malay.sh`, `eval_qwen3_asr_0_6b_malay_sft_fleurs.sh`, `train_qwen3_asr_0_6b_malay_sft_lowlr_bs2_extra2ep.sh`, and `eval_qwen3_asr_0_6b_malay_sft_lowlr_bs2_extra2ep_fleurs.sh`.
- English/Chinese pretrained baselines: `eval_qwen3_asr_0_6b_pretrained_fleurs_english.sh` and `eval_qwen3_asr_0_6b_pretrained_fleurs_chinese.sh`.
- Shared wrappers: `eval_qwen3_asr_manifest.py`, `eval_qwen3_asr_manifest_sbatch.sh`, `eval_qwen3_asr_manifest_qwen3_asr_env_sbatch.sh`, `train_qwen3_asr_sft_sbatch.sh`, `eval_meralion3_asr_manifest.py`, `eval_meralion3_asr_manifest_sbatch.sh`, and `compute_mixed_error_rate.py`.

Processed Tamil training data is in `/mnt/weka/aisg/speech_spoke/donghang/data/deepdml-iisc-mile-tamil-asr`; FLEURS Tamil test data is in `/mnt/weka/aisg/speech_spoke/donghang/data/fleur/tamil`. Processed Malay FLEURS data is in `/mnt/weka/aisg/speech_spoke/donghang/data/fleur/malay`. Preserve `raw_transcription` in generated manifests.

Recorded metrics are tracked in `finetuning_tamil_todo.md`, `finetuning_malay_todo.md`, and output `metrics.json` files. Tamil FLEURS results: 0.6B pretrained WER/CER `1.3438/1.0193`; 0.6B Tamil SFT checkpoint `outputs/qwen3-asr-0.6b-tamil-sft/checkpoint-2417` scored `0.4517/0.1611`; 0.6B code-switch SFT scored `0.6433/0.2433`; 1.7B pretrained scored `1.5006/1.0421`; 1.7B Tamil SFT scored `0.6160/0.2499`; 1.7B code-switch SFT scored `0.6359/0.2597`. Malay FLEURS: 0.6B pretrained `0.1753/0.0541`; initial Malay SFT `checkpoint-84` `0.2341/0.0900`; low-LR bs2 extra-2-epoch continuation `checkpoint-334` `0.2036/0.0763`. English/Chinese FLEURS pretrained: 0.6B English `0.0545/0.0226`, Chinese `0.0462/0.0515`; 1.7B English `0.0439/0.0170`, Chinese `0.0425/0.0489`.

## MERaLiON And Code-Switching Context

MERaLiON NSC v1 was extracted from `/mnt/weka/aisg/speech_spoke/donghang/hf_cache/hub/datasets--MERaLiON--Multitask-National-Speech-Corpus-v1/snapshots/d169debbce4a2dc07f7293360d3eed33c06793b9` into `/mnt/weka/aisg/speech_spoke/donghang/data/meralion-multitask-national-speech-corpus-v1`. Use `scripts/prepare_meralion_nsc_asr.py` for reproducible extraction and `scripts/prepare_meralion_nsc_asr_sbatch.sh` for the Slurm array wrapper. Parts 1-3 are English ASR with Singlish accent; part 4 is Chinese-English, Tamil-English, and Malay-English code-switching. Final extracted row counts: part001 `2,258,301/3,000`, part002 `2,473,990/3,000`, part003 `96,245/1,000`, and part004 `9,629/1,000` for train/test.

The current code-switching training manifest is `/mnt/weka/aisg/speech_spoke/donghang/data/code-switching/train_tamil_part1_3_45000_part4.jsonl`. It was created by `scripts/build_code_switching_train_manifest.py` with seed `20260802`, using 15,000 sampled rows from each of MERaLiON part1-3, all part4 train rows, and all DeepDML Tamil train rows for a total of 131,943 rows. The code-switching training wrapper is `scripts/train_qwen3_asr_0_6b_tamil_singlish_codeswitch_sft.sh`, outputting to `outputs/qwen3-asr-0.6b-tamil-singlish-codeswitch-sft`.

Part4 evaluation uses `/mnt/weka/aisg/speech_spoke/donghang/data/meralion-multitask-national-speech-corpus-v1/test_part004.jsonl`, not train data. Do not force a single language for part4 eval because the references are code-switching. Do not force `language=Tamil`; the Qwen wrapper rejects Tamil in `validate_language`, and earlier successful Tamil runs used automatic language. `eval_qwen3_asr_manifest.py` supports `--remove_speaker_tags`, `--remove_unicode_punct`, and `--cjk_char_space` for mixed-script normalization.

The initial code-switching train job failed because step eval was enabled without an eval dataset; `finetuning/qwen3_asr_sft.py` was fixed to set `eval_strategy="no"` when `--eval_file` is omitted. The code-switching training manifest is known to contain polluted `text` targets such as `language Tamil<asr_text>...` and speaker tags. The SFT script trained from `ex["text"]`, so code-switch SFT results are worse than pretrained. If improving this direction, rebuild a clean manifest from `transcription` or `raw_transcription`, strip or consistently handle speaker tags, and upweight part4.

## Current Part4 Results And MER

MER is computed by `scripts/compute_mixed_error_rate.py`: Chinese and Tamil are scored as characters; English and Malay Latin-script spans are scored as whitespace words. Full MERaLiON part4 test has 1000 samples. A secondary filtered report excludes the 144 samples whose reference contains any Tamil Unicode character, leaving 856 samples.

Full part4 results:

| Model | WER | CER | MER | Output |
| --- | ---: | ---: | ---: | --- |
| 0.6B pretrained | `0.4454` | `0.3235` | `0.4764` | `outputs/meralion_part4_eval_pretrained_0_6b` |
| 0.6B code-switch SFT | `0.5340` | `0.4203` | `0.5570` | `outputs/meralion_part4_eval_codeswitch_sft_0_6b` |
| 1.7B pretrained | `0.4157` | `0.3003` | `0.4522` | `outputs/meralion_part4_eval_pretrained_1_7b` |
| 1.7B code-switch SFT | `0.4737` | `0.3582` | `0.4990` | `outputs/meralion_part4_eval_codeswitch_sft_1_7b` |
| Polyglot-Lion-1.7B | `0.4316` | `0.3138` | `0.4630` | `outputs/meralion_part4_eval_polyglot_lion_1_7b_alt_env` |
| MERaLiON-3-ASR | `0.2665` | `0.1833` | `0.2613` | `outputs/meralion_part4_eval_meralion3_3b_asr` |
| 0.6B Tamil-only SFT | `1.0033` | `0.9377` | not computed | `outputs/meralion_part4_eval_tamil_sft_0_6b` |

Part4 results after excluding Tamil-reference samples:

| Model | WER | CER | MER | Output file |
| --- | ---: | ---: | ---: | --- |
| 0.6B pretrained | `0.3985` | `0.2922` | `0.3985` | `metrics_no_tamil_reference.json` |
| 0.6B code-switch SFT | `0.4925` | `0.3895` | `0.4925` | `metrics_no_tamil_reference.json` |
| 1.7B pretrained | `0.3703` | `0.2687` | `0.3703` | `metrics_no_tamil_reference.json` |
| 1.7B code-switch SFT | `0.4259` | `0.3260` | `0.4259` | `metrics_no_tamil_reference.json` |
| Polyglot-Lion-1.7B | `0.3899` | `0.2858` | `0.3899` | `metrics_no_tamil_reference.json` |
| MERaLiON-3-ASR | `0.2287` | `0.1672` | `0.2294` | `metrics_no_tamil_reference.json` |

MERaLiON part1 English/Singlish results: 0.6B pretrained WER/CER `0.0749/0.0353`; 0.6B code-switch SFT `0.0788/0.0378`; 1.7B pretrained `0.0625/0.0290`; 1.7B code-switch SFT `0.0568/0.0274`.

## Coding Style & Naming Conventions

Follow existing Python style: 4-space indentation, type hints for public helpers where practical, dataclasses for structured results, and concise docstrings for user-facing classes or CLI modules. Keep model-facing names explicit, such as `Qwen3ASRModel`, `ASRTranscription`, and `Qwen3ForcedAligner`. Preserve Apache-2.0 headers on source files that already include them. Avoid broad `except` blocks in new code unless guarding optional dependencies such as vLLM.

## Testing Guidelines

No dedicated `tests/` tree is currently checked in. For changes, add focused tests when introducing deterministic utility behavior, preferably under `tests/` with names like `test_audio_utils.py`. For model or GPU paths, document the exact smoke command used and prefer small public audio examples from the README. Verify both transformers and vLLM paths when touching shared inference APIs.

## Commit & Pull Request Guidelines

Recent history uses short imperative or descriptive messages, for example `fix streaming unicode` and `Add news on Transformers native support`. Keep commits focused and mention the affected area, such as `cli: fix streaming unicode parsing`. Pull requests should include a short problem statement, implementation summary, commands run, and any model/backend assumptions. Include screenshots only for Gradio UI changes.

## Security & Configuration Tips

Do not commit model weights, generated checkpoints, API keys, or large audio corpora. Keep local paths, CUDA device choices, and downloaded model locations configurable through CLI arguments or environment variables.
