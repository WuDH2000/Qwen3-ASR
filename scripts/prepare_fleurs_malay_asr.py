#!/usr/bin/env python3
# coding=utf-8
"""Convert local FLEURS Malay cache to Qwen3-ASR JSONL manifests and WAV files."""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List

import soundfile as sf
from datasets import Audio, Dataset


DEFAULT_SOURCE = (
    "/mnt/weka/aisg/speech_spoke/donghang/hf_cache/datasets/google___fleurs/"
    "ms_my/0.0.0/70bb2e84b976b7e960aa89f1c648e09c59f894dd"
)
DEFAULT_OUTPUT = "/mnt/weka/aisg/speech_spoke/donghang/data/fleur/malay"


def split_files(source_dir: Path, split: str) -> List[Path]:
    if split == "train":
        return sorted(source_dir.glob("fleurs-train-*.arrow"))
    return [source_dir / f"fleurs-{split}.arrow"]


def safe_name(sample_id: object, audio_path: str, fallback: str) -> str:
    stem = Path(audio_path or "").stem or str(sample_id or fallback)
    stem = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in stem)
    return f"{stem}.wav"


def iter_rows(source_dir: Path, split: str) -> Iterable[Dict[str, object]]:
    for arrow_path in split_files(source_dir, split):
        ds = Dataset.from_file(str(arrow_path)).cast_column("audio", Audio(decode=False))
        for row in ds:
            yield row


def convert_split(source_dir: Path, output_dir: Path, split: str, jsonl_name: str) -> Dict[str, int]:
    audio_dir = output_dir / "audio" / split
    audio_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / jsonl_name

    written = 0
    skipped = 0
    validated = 0

    with out_path.open("w", encoding="utf-8") as out_f:
        for idx, row in enumerate(iter_rows(source_dir, split)):
            audio = row.get("audio") or {}
            audio_bytes = audio.get("bytes")
            transcription = (row.get("transcription") or "").strip()
            raw_transcription = (row.get("raw_transcription") or transcription).strip()
            if not audio_bytes or not transcription:
                skipped += 1
                continue

            wav_path = audio_dir / safe_name(row.get("id"), audio.get("path") or row.get("path"), f"{split}_{idx:06d}")
            if not wav_path.exists():
                wav_path.write_bytes(audio_bytes)

            if validated < 10:
                info = sf.info(str(wav_path))
                if info.samplerate != 16000 or info.channels != 1:
                    raise ValueError(f"Unexpected audio format for {wav_path}: {info}")
                validated += 1

            item = {
                "audio": str(wav_path),
                "audio_path": str(wav_path),
                "text": f"language Malay<asr_text>{transcription}",
                "transcription": transcription,
                "raw_transcription": raw_transcription,
                "id": str(row.get("id", idx)),
                "language": row.get("language", "Malay"),
                "num_samples": row.get("num_samples"),
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
    split_map = {
        "train": "train.jsonl",
        "validation": "validation.jsonl",
        "test": "test.jsonl",
    }
    stats: Dict[str, Dict[str, int]] = {}
    for split, jsonl_name in split_map.items():
        stats[split] = convert_split(source_dir, output_dir, split, jsonl_name)

    metadata = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_dir": str(source_dir),
        "output_dir": str(output_dir),
        "language_prefix": "language Malay<asr_text>",
        "training_text_source": "transcription",
        "raw_transcription_preserved": True,
        "splits": {
            split: {
                "jsonl": str(output_dir / jsonl_name),
                "rows": count_jsonl(output_dir / jsonl_name),
                **stats[split],
            }
            for split, jsonl_name in split_map.items()
        },
    }
    with (output_dir / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    with (output_dir / "conversion_report.txt").open("w", encoding="utf-8") as f:
        for split, jsonl_name in split_map.items():
            f.write(f"{split}_jsonl={output_dir / jsonl_name}\n")
            f.write(f"{split}_rows={count_jsonl(output_dir / jsonl_name)}\n")
            f.write(f"{split}_skipped={stats[split]['skipped']}\n")

    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
