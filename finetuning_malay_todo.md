# Qwen3-ASR 0.6B Malay Fine-tuning To-Do

## Goal

Fine-tune local `Qwen3-ASR-0.6B` on local FLEURS Malay (`ms_my`) data, save processed manifests/audio under `/mnt/weka/aisg/speech_spoke/donghang/data/fleur/malay`, run Slurm-only training on the allowed `smc-pod` nodes, and compute WER/CER for both pretrained and fine-tuned checkpoints.

## To-Do List

| ID | Task | Validation Plan | Passing Result | Status |
| --- | --- | --- | --- | --- |
| M1 | Inspect FLEURS Malay cache. | Check local cache structure, splits, fields, and row counts. | Found train/validation/test arrow files for `ms_my`. | Complete |
| M2 | Convert data to Qwen3-ASR manifests and WAV files. | Generate `train.jsonl`, `validation.jsonl`, `test.jsonl`; verify line counts, audio existence, and 16 kHz mono WAV format. | 2667 train, 324 validation, 749 test rows; 3740 WAV files; zero missing audio. | Complete |
| M3 | Create Slurm fine-tune and eval scripts. | Run `bash -n`, `python -m py_compile`, and eval dry-run. | Scripts are syntactically valid and FLEURS test dry-run passes. | Complete |
| M4 | Submit Malay fine-tune. | Submit with `sbatch`; inspect Slurm status, logs, and checkpoints. | Job completed with exit code `0:0`; checkpoint exists. | Complete |
| M5 | Run pretrained baseline eval. | Submit pretrained 0.6B eval; verify predictions and metrics. | Job completed with predictions and WER/CER. | Complete |
| M6 | Run fine-tuned checkpoint eval. | Submit latest Malay SFT checkpoint eval; verify predictions and metrics. | Job completed with predictions and WER/CER. | Complete |
| M7 | Run lower-LR smaller-batch continuation. | Start from `checkpoint-84`, lower LR, lower per-device batch size, train 2 more epochs, then evaluate via dependent Slurm job. | Job completed with exit code `0:0`; `checkpoint-334` exists; dependent eval wrote WER/CER. | Complete |

## Artifacts

- Data dir: `/mnt/weka/aisg/speech_spoke/donghang/data/fleur/malay`
- Manifests: `train.jsonl`, `validation.jsonl`, `test.jsonl`
- Data conversion script: `scripts/prepare_fleurs_malay_asr.py`
- Generic eval script: `scripts/eval_qwen3_asr_manifest.py`
- Fine-tune script: `scripts/train_qwen3_asr_0_6b_malay_sft.sh`
- Pretrained eval script: `scripts/eval_qwen3_asr_0_6b_pretrained_fleurs_malay.sh`
- Fine-tuned eval script: `scripts/eval_qwen3_asr_0_6b_malay_sft_fleurs.sh`
- Fine-tuned checkpoint: `outputs/qwen3-asr-0.6b-malay-sft/checkpoint-84`
- Pretrained eval output: `outputs/fleurs_malay_eval_pretrained_0_6b`
- Fine-tuned eval output: `outputs/fleurs_malay_eval_finetuned_0_6b`
- Lower-LR continuation script: `scripts/train_qwen3_asr_0_6b_malay_sft_lowlr_bs2_extra2ep.sh`
- Lower-LR continuation eval script: `scripts/eval_qwen3_asr_0_6b_malay_sft_lowlr_bs2_extra2ep_fleurs.sh`
- Lower-LR continuation checkpoint: `outputs/qwen3-asr-0.6b-malay-sft-lowlr-bs2-extra2ep/checkpoint-334`
- Lower-LR continuation eval output: `outputs/fleurs_malay_eval_finetuned_0_6b_lowlr_bs2_extra2ep`

## Results

| Model | Job ID | Samples | WER | CER | Notes |
| --- | ---: | ---: | ---: | ---: | --- |
| Pretrained Qwen3-ASR-0.6B | 168779 | 749 | 0.17533944448273486 | 0.054131720194433894 | Forced `language=Malay` |
| Fine-tuned checkpoint-84 | 168780 | 749 | 0.2341305396650355 | 0.08995976893928324 | Forced `language=Malay` |
| Fine-tuned low-LR checkpoint-334 | 168797 | 749 | 0.2035977669033014 | 0.07629289770790107 | Continued from `checkpoint-84`; forced `language=Malay` |

## Notes

- Source cache: `/mnt/weka/aisg/speech_spoke/donghang/hf_cache/datasets/google___fleurs/ms_my/0.0.0/70bb2e84b976b7e960aa89f1c648e09c59f894dd`
- FLEURS splits: train 2667, validation 324, test 749.
- Training uses `transcription` as the target text: `language Malay<asr_text>{transcription}`.
- `raw_transcription` is preserved in every generated manifest row.
- Metrics normalization: strip, remove zero-width chars, collapse whitespace, lowercase, remove ASCII punctuation.
- Slurm jobs ran on `smc-pod-4` within the allowed node list.
- Continuation training job `168796` completed on `smc-pod-4` with exit code `0:0`. It initialized from `outputs/qwen3-asr-0.6b-malay-sft/checkpoint-84`, used `batch_size=2`, `grad_acc=8`, `lr=5e-6`, `epochs=2`, `save_steps=100`, and wrote checkpoints through `checkpoint-334`.
- Dependent continuation eval job `168797` completed on `smc-pod-4` with exit code `0:0`. It evaluated `outputs/qwen3-asr-0.6b-malay-sft-lowlr-bs2-extra2ep/checkpoint-334` and wrote metrics/predictions to `outputs/fleurs_malay_eval_finetuned_0_6b_lowlr_bs2_extra2ep`.
- Lower-LR continuation improved over the first fine-tuned checkpoint (`WER 0.2341 -> 0.2036`, `CER 0.0900 -> 0.0763`) but remains worse than the pretrained baseline (`WER 0.1753`, `CER 0.0541`).

## Follow-up Improvements

- Inspect error cases in `outputs/fleurs_malay_eval_finetuned_0_6b_lowlr_bs2_extra2ep/predictions.jsonl` against the pretrained baseline to identify whether fine-tuning is causing deletions, insertions, or spelling drift.
- Try validation-based checkpoint selection instead of always evaluating the latest checkpoint.
- Consider freezing more of the model or using a smaller LR such as `1e-6` because Malay FLEURS is small and pretrained performance is already strong.
