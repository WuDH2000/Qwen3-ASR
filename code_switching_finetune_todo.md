# Qwen3-ASR Code-Switching Fine-Tuning To-Do

## Goal

Fine-tune Qwen3-ASR-0.6B for Tamil plus Singlish/code-switching ASR using all DeepDML Tamil train data, all MERaLiON part4 train data, and 15,000 sampled train rows from each MERaLiON English part 1-3. Evaluate pretrained and fine-tuned checkpoints on MERaLiON part4 test with WER/CER.

## To-Do List

| ID | Task | Validation Plan | Passing Result | Status |
| --- | --- | --- | --- | --- |
| C1 | Build combined train manifest. | Sample 15,000 rows from each part1-3 with fixed seed, append part4 and Tamil, shuffle, and verify counts/audio paths. | `train_tamil_part1_3_45000_part4.jsonl` has 131,943 rows and zero sampled missing audio. | Complete |
| C2 | Create Slurm fine-tune script. | Run `bash -n`; verify model path, train file, output dir, env, and node allowlist. | Script is syntax valid and uses only allowed nodes. | Complete |
| C3 | Create part4 pretrained eval script. | Run `bash -n` and eval dry-run. | Script evaluates local pretrained 0.6B on part4 test without forcing one language. | Complete |
| C4 | Submit pretrained part4 eval. | Submit with `sbatch`; inspect status, logs, predictions, and metrics. | Job completes with `metrics.json` and `predictions.jsonl`. | Complete |
| C5 | Submit code-switching fine-tune. | Submit with `sbatch`; inspect logs and checkpoints. | Job completes and writes inferable `checkpoint-8247`. | Complete |
| C6 | Evaluate fine-tuned checkpoint on part4 test. | Submit eval wrapper using latest checkpoint without forcing one language. | Job writes part4 WER/CER for the fine-tuned checkpoint. | Submitted as `169169` |
| C7 | Evaluate fine-tuned checkpoint on Tamil FLEURS test. | Submit eval wrapper using latest checkpoint with `LANGUAGE=Tamil`. | Job writes Tamil WER/CER for the fine-tuned checkpoint. | Submitted as `169170` |
| C8 | Evaluate fine-tuned checkpoint on MERaLiON part1 test. | Submit eval wrapper using latest checkpoint with `LANGUAGE=English`. | Job writes part1 WER/CER for the fine-tuned checkpoint. | Submitted as `169168` |

## Notes

- Tamil train: `/mnt/weka/aisg/speech_spoke/donghang/data/deepdml-iisc-mile-tamil-asr/train.jsonl`
- MERaLiON dir: `/mnt/weka/aisg/speech_spoke/donghang/data/meralion-multitask-national-speech-corpus-v1`
- Part4 test: `/mnt/weka/aisg/speech_spoke/donghang/data/meralion-multitask-national-speech-corpus-v1/test_part004.jsonl`
- Part4 eval should not force a single language because the data is code-switching.
- Combined train manifest: `/mnt/weka/aisg/speech_spoke/donghang/data/code-switching/train_tamil_part1_3_45000_part4.jsonl`
- Combined manifest seed: `20260802`.
- Combined source counts: part001 sampled 15,000; part002 sampled 15,000; part003 sampled 15,000; part004 all 9,629; Tamil all 77,314; total 131,943.
- Added scripts: `build_code_switching_train_manifest.py`, `train_qwen3_asr_0_6b_tamil_singlish_codeswitch_sft.sh`, `eval_qwen3_asr_0_6b_pretrained_meralion_part4.sh`, and `eval_qwen3_asr_0_6b_codeswitch_sft_meralion_part4.sh`.
- Part4 eval dry-run passed: 1,000 samples, zero missing audio, zero empty references.
- Submitted pretrained part4 eval as job `168842`; it completed and writes to `outputs/meralion_part4_eval_pretrained_0_6b` with WER `0.44538503171012644` and CER `0.32348423571321877`.
- Submitted code-switching fine-tune as job `168843`; it failed before training because `qwen3_asr_sft.py` enabled step eval without an eval dataset. Fixed the training entrypoint to set `eval_strategy="no"` when `--eval_file` is omitted.
- Resubmitted code-switching fine-tune as job `168844`; it completed on `smc-pod-4` and wrote latest checkpoint `outputs/qwen3-asr-0.6b-tamil-singlish-codeswitch-sft/checkpoint-8247`.
- Pretrained 0.6B MERaLiON part1 eval job `168864` completed at `outputs/meralion_part1_eval_pretrained_0_6b` with WER `0.07486974343778552` and CER `0.03527731114667795`.
- Submitted code-switching SFT checkpoint evals from `checkpoint-8247`: MERaLiON part1 job `169168` to `outputs/meralion_part1_eval_codeswitch_sft_0_6b`; MERaLiON part4 job `169169` to `outputs/meralion_part4_eval_codeswitch_sft_0_6b`; Tamil FLEURS job `169170` to `outputs/fleurs_tamil_eval_codeswitch_sft_0_6b`. All were pending due to Slurm priority at submission check.
