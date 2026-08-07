# MERaLiON NSC ASR Data Preparation To-Do

## Goal

Extract ASR parts 1-4 from the local MERaLiON Multitask National Speech Corpus v1 cache into per-part train/test JSONL manifests and local audio files for Qwen3-ASR workflows.

## To-Do List

| ID | Task | Validation Plan | Passing Result | Status |
| --- | --- | --- | --- | --- |
| N1 | Inspect source cache. | Confirm snapshot directories, parquet schema, audio bytes, and train/test splits for parts 1-4. | `ASR-PART1..4-Train/Test` exist; schema has `context`, `instruction`, `answer`; audio bytes are WAV. | Complete |
| N2 | Create conversion script. | Run syntax check and small part/split conversion. | Script writes expected JSONL fields and validates sampled 16 kHz mono audio. | Complete |
| N3 | Extract all parts. | Run converter for parts 1-4; verify row counts and audio file counts. | `train_part001..004.jsonl` and `test_part001..004.jsonl` exist with matching audio. | Complete |
| N4 | Record final artifacts. | Update this file with output path, row counts, skipped counts, and validation results. | Preparation status is reproducible from this file. | Complete |

## Notes

- Source cache: `/mnt/weka/aisg/speech_spoke/donghang/hf_cache/hub/datasets--MERaLiON--Multitask-National-Speech-Corpus-v1/snapshots/d169debbce4a2dc07f7293360d3eed33c06793b9`
- Output dir: `/mnt/weka/aisg/speech_spoke/donghang/data/meralion-multitask-national-speech-corpus-v1`
- Parts 1-3 are English ASR. Part 4 contains Chinese-English, Tamil-English, and Malay-English code-switching data.
- Manifest rows preserve `raw_transcription`, `transcription`, `prompt`, and `source_instruction`.
- Added `scripts/prepare_meralion_nsc_asr.py` and `scripts/prepare_meralion_nsc_asr_sbatch.sh`.
- Smoke test output under `outputs/meralion_nsc_smoke` wrote 2 train and 2 test rows for part 4; sampled audio validated as 16 kHz mono WAV.
- Submitted Slurm array job `168805` for parts 1-4. Part 4 completed in 18s with 9,629 train rows and 1,000 test rows. Part 3 completed in 3m25s with 96,245 train rows and 1,000 test rows. Both skipped 0 rows.
- Slurm array job `168805` completed all tasks with exit code `0:0`. Part 1 completed in 28m58s; part 2 completed in 28m57s; both ran on `smc-pod-4`.

## Final Artifacts

| Part | Train JSONL Rows | Test JSONL Rows | Skipped | Metadata |
| --- | ---: | ---: | ---: | --- |
| part001 | 2,258,301 | 3,000 | 0 | `/mnt/weka/aisg/speech_spoke/donghang/data/meralion-multitask-national-speech-corpus-v1/metadata_part001.json` |
| part002 | 2,473,990 | 3,000 | 0 | `/mnt/weka/aisg/speech_spoke/donghang/data/meralion-multitask-national-speech-corpus-v1/metadata_part002.json` |
| part003 | 96,245 | 1,000 | 0 | `/mnt/weka/aisg/speech_spoke/donghang/data/meralion-multitask-national-speech-corpus-v1/metadata_part003.json` |
| part004 | 9,629 | 1,000 | 0 | `/mnt/weka/aisg/speech_spoke/donghang/data/meralion-multitask-national-speech-corpus-v1/metadata_part004.json` |

Total JSONL rows: 4,846,165. Output manifests are under `/mnt/weka/aisg/speech_spoke/donghang/data/meralion-multitask-national-speech-corpus-v1` with names `train_part001.jsonl` through `train_part004.jsonl` and `test_part001.jsonl` through `test_part004.jsonl`.

Validation: `wc -l` row counts match metadata. First-row audio samples from all 8 manifests exist and were read by `soundfile` as 16 kHz mono WAV. Manifest rows include `audio`, `audio_path`, `text`, `transcription`, `raw_transcription`, `prompt`, `source_instruction`, `source_parquet`, and `source_context_path`.
