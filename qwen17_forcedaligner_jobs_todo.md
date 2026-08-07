# Qwen3-ASR 1.7B And ForcedAligner 0.6B Jobs

## Goal

Submit Tamil/code-switching SFT jobs for Qwen3-ASR-1.7B and Qwen3-ForcedAligner-0.6B, plus pretrained WER/CER evals on Tamil, Malay, Chinese, English, and MERaLiON part1 test.

## Job Status

| Job ID | Name | Task | Output |
| ---: | --- | --- | --- |
| 168937 | `qwen17_tamil_sft` | Qwen3-ASR-1.7B Tamil SFT | `outputs/qwen3-asr-1.7b-tamil-sft` |
| 168938 | `qwen17_codesw_sft` | Qwen3-ASR-1.7B code-switching SFT | `outputs/qwen3-asr-1.7b-tamil-singlish-codeswitch-sft` |
| 168939 | `qwenfa06_tamil_sft` | Failed: ForcedAligner classifier head shape mismatch during ASR SFT | `outputs/qwen3-forcedaligner-0.6b-tamil-sft` |
| 168940 | `qwenfa06_codesw_sft` | Failed: ForcedAligner classifier head shape mismatch during ASR SFT | `outputs/qwen3-forcedaligner-0.6b-tamil-singlish-codeswitch-sft` |
| 169154 | `qwenfa06_smoke_asrhead` | Smoke test: ForcedAligner loaded with ASR vocab head, `max_steps=1` | `outputs/qwen3-forcedaligner-0.6b-tamil-sft-smoke-asrhead` |
| 169158 | `qwenfa06_tamil_sft_fix` | Resubmitted after `afterok:169154`: ForcedAligner-0.6B Tamil ASR SFT with ASR vocab head | `outputs/qwen3-forcedaligner-0.6b-tamil-sft-asrhead` |
| 169159 | `qwenfa06_codesw_sft_fix` | Resubmitted after `afterok:169154`: ForcedAligner-0.6B code-switching ASR SFT with ASR vocab head | `outputs/qwen3-forcedaligner-0.6b-tamil-singlish-codeswitch-sft-asrhead` |
| 168941 | `qwen17_ta_base_eval` | Qwen3-ASR-1.7B FLEURS Tamil pretrained eval | `outputs/fleurs_tamil_eval_pretrained_1_7b` |
| 168942 | `qwen17_ms_base_eval` | Qwen3-ASR-1.7B FLEURS Malay pretrained eval | `outputs/fleurs_malay_eval_pretrained_1_7b` |
| 168943 | `qwen17_zh_base_eval` | Qwen3-ASR-1.7B FLEURS Chinese pretrained eval | `outputs/fleurs_chinese_eval_pretrained_1_7b` |
| 168944 | `qwen17_en_base_eval` | Qwen3-ASR-1.7B FLEURS English pretrained eval | `outputs/fleurs_english_eval_pretrained_1_7b` |
| 168945 | `qwen17_p1_base_eval` | Qwen3-ASR-1.7B MERaLiON part1 pretrained eval | `outputs/meralion_part1_eval_pretrained_1_7b` |
| 169197 | `qwen17_tamil_sft_eval` | Qwen3-ASR-1.7B Tamil SFT checkpoint-4833 FLEURS Tamil eval | `outputs/fleurs_tamil_eval_finetuned_1_7b` |
| 169305 | `qwen17_cs_ta_eval` | Qwen3-ASR-1.7B code-switching checkpoint-8247 FLEURS Tamil eval | `outputs/fleurs_tamil_eval_codeswitch_sft_1_7b` |
| 169306 | `qwen17_cs_p1_eval` | Qwen3-ASR-1.7B code-switching checkpoint-8247 MERaLiON part1 eval | `outputs/meralion_part1_eval_codeswitch_sft_1_7b` |
| 169304 | `qwen17_cs_p4_eval` | Qwen3-ASR-1.7B code-switching checkpoint-8247 MERaLiON part4 eval | `outputs/meralion_part4_eval_codeswitch_sft_1_7b` |
| 168946 | `qwenfa06_ta_base_eval` | ForcedAligner-0.6B FLEURS Tamil pretrained eval | `outputs/fleurs_tamil_eval_pretrained_forcedaligner_0_6b` |
| 168947 | `qwenfa06_ms_base_eval` | ForcedAligner-0.6B FLEURS Malay pretrained eval | `outputs/fleurs_malay_eval_pretrained_forcedaligner_0_6b` |
| 168948 | `qwenfa06_zh_base_eval` | ForcedAligner-0.6B FLEURS Chinese pretrained eval | `outputs/fleurs_chinese_eval_pretrained_forcedaligner_0_6b` |
| 168949 | `qwenfa06_en_base_eval` | ForcedAligner-0.6B FLEURS English pretrained eval | `outputs/fleurs_english_eval_pretrained_forcedaligner_0_6b` |
| 168950 | `qwenfa06_p1_base_eval` | ForcedAligner-0.6B MERaLiON part1 pretrained eval | `outputs/meralion_part1_eval_pretrained_forcedaligner_0_6b` |

## Notes

- Generic wrappers: `scripts/train_qwen3_asr_sft_sbatch.sh` and `scripts/eval_qwen3_asr_manifest_sbatch.sh`.
- All jobs were submitted with `#SBATCH --nodelist=smc-pod-3,smc-pod-4,smc-pod-5,smc-pod-8,smc-pod-9,smc-pod-10`.
- Qwen3-ASR-1.7B snapshot: `/mnt/weka/aisg/speech_spoke/donghang/hf_cache/hub/models--Qwen--Qwen3-ASR-1.7B/snapshots/7278e1e70fe206f11671096ffdd38061171dd6e5`.
- Qwen3-ASR-1.7B Tamil SFT job `168937` completed and wrote latest checkpoint `outputs/qwen3-asr-1.7b-tamil-sft/checkpoint-4833`; Tamil FLEURS eval was submitted as job `169197` and was `PENDING (Priority)` at submission check.
- Qwen3-ASR-1.7B code-switching SFT job `168938` completed and wrote latest checkpoint `outputs/qwen3-asr-1.7b-tamil-singlish-codeswitch-sft/checkpoint-8247`; evals were submitted on 2026-08-03 as Tamil FLEURS job `169305`, MERaLiON part1 job `169306`, and MERaLiON part4 job `169304`. All were `PENDING (Priority)` at submission check.
- ForcedAligner-0.6B snapshot: `/mnt/weka/aisg/speech_spoke/donghang/hf_cache/hub/models--Qwen--Qwen3-ForcedAligner-0.6B/snapshots/c7cbfc2048c462b0d63a45797104fc9db3ad62b7`.
- ForcedAligner-0.6B supports Chinese, Cantonese, English, German, Spanish, French, Italian, Portuguese, Russian, Korean, and Japanese. Tamil/Malay evals leave `LANGUAGE` unset to avoid unsupported-language validation failures.
- ForcedAligner ASR SFT fix: `thinker.lm_head.weight` in the checkpoint is `[5000, 1024]`, while `thinker.model.embed_tokens.weight` is `[152064, 1024]`. `finetuning/qwen3_asr_sft.py` now supports `--force_asr_lm_head`, which loads the checkpoint with `thinker_config.model_type=qwen3_asr`, ties embeddings, and uses `ignore_mismatched_sizes=True` so the 5000-class forced-aligner head is replaced by a full ASR vocab head.
- Validation standard for the fix: job `169154` must complete one train step without the previous `shape '[-1, 152064]' is invalid` error. Jobs `169158` and `169159` are dependency-gated on `afterok:169154`.
- Current Slurm status at fix submission check: smoke job `169154` was `PENDING (Priority)`; resubmitted jobs `169158` and `169159` were submitted with dependency `afterok:169154`.
