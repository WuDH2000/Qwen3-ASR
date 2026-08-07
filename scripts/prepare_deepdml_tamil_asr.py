#!/usr/bin/env python3
# coding=utf-8
"""Convert DeepDML IISC MILE Tamil ASR parquet shards to Qwen3-ASR SFT JSONL."""

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, Tuple

import pyarrow.parquet as pq
import soundfile as sf


DEFAULT_SOURCE = (
    "/mnt/weka/aisg/speech_spoke/donghang/hf_cache/hub/"
    "datasets--deepdml--iisc-mile-tamil-asr/snapshots/"
    "32f434a8e3fe6705304964fffd90dab3383ee6d1"
)
DEFAULT_OUTPUT = "/mnt/weka/aisg/speech_spoke/donghang/data/deepdml-iisc-mile-tamil-asr"


def iter_parquet_files(source_dir: Path, split: str) -> Iterable[Path]:
    data_dir = source_dir / "data"
    yield from sorted(data_dir.glob(f"{split}-*.parquet"))


def split_rows(source_dir: Path, split: str) -> int:
    return sum(pq.ParquetFile(path).metadata.num_rows for path in iter_parquet_files(source_dir, split))


def safe_wav_name(sample_id: str, fallback: str) -> str:
    name = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in (sample_id or fallback))
    return f"{name}.wav"


def convert_split(source_dir: Path, output_dir: Path, split: str, jsonl_name: str) -> Dict[str, int]:
    audio_dir = output_dir / "audio" / split
    audio_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = output_dir / jsonl_name

    written = 0
    skipped = 0
    validated = 0

    with jsonl_path.open("w", encoding="utf-8") as out_f:
        for parquet_path in iter_parquet_files(source_dir, split):
            table = pq.read_table(parquet_path, columns=["audio", "transcription", "id", "duration"])
            for row_idx, row in enumerate(table.to_pylist()):
                sample_id = row.get("id") or f"{parquet_path.stem}_{row_idx:06d}"
                transcription = (row.get("transcription") or "").strip()
                audio = row.get("audio") or {}
                audio_bytes = audio.get("bytes")
                if not transcription or not audio_bytes:
                    skipped += 1
                    continue

                wav_path = audio_dir / safe_wav_name(sample_id, f"{parquet_path.stem}_{row_idx:06d}")
                if not wav_path.exists():
                    wav_path.write_bytes(audio_bytes)

                if validated < 10:
                    info = sf.info(str(wav_path))
                    if info.samplerate != 16000 or info.channels != 1:
                        raise ValueError(f"Unexpected audio format for {wav_path}: {info}")
                    validated += 1

                item = {
                    "audio": str(wav_path),
                    "text": f"language Tamil<asr_text>{transcription}",
                    "prompt": "",
                    "id": sample_id,
                    "duration": row.get("duration"),
                }
                out_f.write(json.dumps(item, ensure_ascii=False) + "\n")
                written += 1

    return {"written": written, "skipped": skipped, "validated_audio": validated}


def count_jsonl(path: Path) -> int:
    with path.open("r", encoding="utf-8") as f:
        return sum(1 for _ in f)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", default=DEFAULT_SOURCE)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    source_dir = Path(args.source_dir)
    output_dir = Path(args.output_dir)
    if not source_dir.exists():
        raise FileNotFoundError(source_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    report: Dict[str, object] = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_dir": str(source_dir),
        "output_dir": str(output_dir),
        "language_prefix": "language Tamil<asr_text>",
        "splits": {},
    }

    train_stats = convert_split(source_dir, output_dir, "train", "train.jsonl")
    test_stats = convert_split(source_dir, output_dir, "test", "eval.jsonl")

    report["splits"] = {
        "train": {
            "source_rows": split_rows(source_dir, "train"),
            "jsonl": str(output_dir / "train.jsonl"),
            **train_stats,
        },
        "eval": {
            "source_split": "test",
            "source_rows": split_rows(source_dir, "test"),
            "jsonl": str(output_dir / "eval.jsonl"),
            **test_stats,
        },
    }

    with (output_dir / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    with (output_dir / "conversion_report.txt").open("w", encoding="utf-8") as f:
        f.write(f"source_dir={source_dir}\n")
        f.write(f"output_dir={output_dir}\n")
        f.write(f"train_jsonl_rows={count_jsonl(output_dir / 'train.jsonl')}\n")
        f.write(f"eval_jsonl_rows={count_jsonl(output_dir / 'eval.jsonl')}\n")
        f.write(f"train_skipped={train_stats['skipped']}\n")
        f.write(f"eval_skipped={test_stats['skipped']}\n")

    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
