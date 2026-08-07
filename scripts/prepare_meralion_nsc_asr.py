#!/usr/bin/env python3
# coding=utf-8
"""Extract MERaLiON Multitask National Speech Corpus ASR parts to JSONL and WAV."""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List

import pyarrow.parquet as pq
import soundfile as sf


DEFAULT_SOURCE = (
    "/mnt/weka/aisg/speech_spoke/donghang/hf_cache/hub/"
    "datasets--MERaLiON--Multitask-National-Speech-Corpus-v1/snapshots/"
    "d169debbce4a2dc07f7293360d3eed33c06793b9"
)
DEFAULT_OUTPUT = "/mnt/weka/aisg/speech_spoke/donghang/data/meralion-multitask-national-speech-corpus-v1"


def iter_parquet_files(source_dir: Path, part: int, split: str) -> Iterable[Path]:
    split_dir = source_dir / f"ASR-PART{part}-{split.capitalize()}"
    if not split_dir.exists():
        raise FileNotFoundError(split_dir)
    yield from sorted(split_dir.glob("*.parquet"))


def source_rows(source_dir: Path, part: int, split: str) -> int:
    return sum(pq.ParquetFile(path).metadata.num_rows for path in iter_parquet_files(source_dir, part, split))


def language_prefix_for_part(part: int) -> str:
    # Parts 1-3 are English. Part 4 contains English code-switching data; keep
    # an English ASR prefix for Qwen SFT compatibility and preserve raw fields.
    return "language English<asr_text>"


def audio_extension(audio_bytes: bytes) -> str:
    if audio_bytes.startswith(b"RIFF") and audio_bytes[8:12] == b"WAVE":
        return ".wav"
    if audio_bytes.startswith(b"fLaC"):
        return ".flac"
    if audio_bytes.startswith(b"OggS"):
        return ".ogg"
    return ".audio"


def make_audio_path(output_dir: Path, part: int, split: str, shard_idx: int, row_idx: int, audio_bytes: bytes) -> Path:
    return (
        output_dir
        / "audio"
        / f"part{part:03d}"
        / split
        / f"part{part:03d}_{split}_{shard_idx:05d}_{row_idx:06d}{audio_extension(audio_bytes)}"
    )


def iter_rows(parquet_path: Path, batch_size: int) -> Iterable[Dict[str, object]]:
    pf = pq.ParquetFile(parquet_path)
    for batch in pf.iter_batches(batch_size=batch_size, columns=["context", "instruction", "answer"]):
        for row in batch.to_pylist():
            yield row


def convert_split(
    source_dir: Path,
    output_dir: Path,
    part: int,
    split: str,
    batch_size: int,
    max_rows: int = 0,
) -> Dict[str, int]:
    audio_dir = output_dir / "audio" / f"part{part:03d}" / split
    audio_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = output_dir / f"{split}_part{part:03d}.jsonl"

    prefix = language_prefix_for_part(part)
    written = 0
    skipped = 0
    validated = 0
    total_bytes = 0

    with jsonl_path.open("w", encoding="utf-8") as out_f:
        for shard_idx, parquet_path in enumerate(iter_parquet_files(source_dir, part, split)):
            for row_idx, row in enumerate(iter_rows(parquet_path, batch_size=batch_size)):
                context = row.get("context") or {}
                audio_bytes = context.get("bytes")
                answer = (row.get("answer") or "").strip()
                instruction = (row.get("instruction") or "").strip()
                if not audio_bytes or not answer:
                    skipped += 1
                    continue

                wav_path = make_audio_path(output_dir, part, split, shard_idx, row_idx, audio_bytes)
                if not wav_path.exists():
                    wav_path.write_bytes(audio_bytes)
                total_bytes += len(audio_bytes)

                if validated < 10:
                    info = sf.info(str(wav_path))
                    if info.samplerate != 16000 or info.channels != 1:
                        raise ValueError(f"Unexpected audio format for {wav_path}: {info}")
                    validated += 1

                sample_id = f"part{part:03d}_{split}_{shard_idx:05d}_{row_idx:06d}"
                item = {
                    "id": sample_id,
                    "part": part,
                    "split": split,
                    "audio": str(wav_path),
                    "audio_path": str(wav_path),
                    "text": f"{prefix}{answer}",
                    "transcription": answer,
                    "raw_transcription": answer,
                    "prompt": instruction,
                    "source_instruction": instruction,
                    "source_parquet": str(parquet_path),
                    "source_context_path": context.get("path"),
                }
                out_f.write(json.dumps(item, ensure_ascii=False) + "\n")
                written += 1
                if max_rows and written >= max_rows:
                    return {
                        "source_rows": source_rows(source_dir, part, split),
                        "written": written,
                        "skipped": skipped,
                        "validated_audio": validated,
                        "audio_bytes": total_bytes,
                        "jsonl": str(jsonl_path),
                        "audio_dir": str(audio_dir),
                        "limited": True,
                    }

    return {
        "source_rows": source_rows(source_dir, part, split),
        "written": written,
        "skipped": skipped,
        "validated_audio": validated,
        "audio_bytes": total_bytes,
        "jsonl": str(jsonl_path),
        "audio_dir": str(audio_dir),
        "limited": False,
    }


def count_jsonl(path: Path) -> int:
    with path.open("r", encoding="utf-8") as f:
        return sum(1 for _ in f)


def parse_parts(raw_parts: str) -> List[int]:
    parts: List[int] = []
    for token in raw_parts.split(","):
        token = token.strip()
        if not token:
            continue
        part = int(token)
        if part < 1 or part > 4:
            raise ValueError(f"Only parts 1-4 are supported for this task: {part}")
        parts.append(part)
    return parts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", default=DEFAULT_SOURCE)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT)
    parser.add_argument("--parts", default="1,2,3,4")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-rows-per-split", type=int, default=0)
    parser.add_argument(
        "--skip-metadata",
        action="store_true",
        help="Do not write aggregate metadata.json; useful when running parts concurrently.",
    )
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
        "text_format": "language English<asr_text>{answer}",
        "raw_transcription_preserved": True,
        "parts": {},
    }

    for part in parse_parts(args.parts):
        report["parts"][f"part{part:03d}"] = {}
        for split in ("train", "test"):
            stats = convert_split(
                source_dir,
                output_dir,
                part,
                split,
                args.batch_size,
                max_rows=args.max_rows_per_split,
            )
            stats["jsonl_rows"] = count_jsonl(Path(stats["jsonl"]))
            report["parts"][f"part{part:03d}"][split] = stats
            print(json.dumps({"part": part, "split": split, **stats}, ensure_ascii=False), flush=True)

        with (output_dir / f"metadata_part{part:03d}.json").open("w", encoding="utf-8") as f:
            json.dump(
                {
                    **{k: v for k, v in report.items() if k != "parts"},
                    "parts": {f"part{part:03d}": report["parts"][f"part{part:03d}"]},
                },
                f,
                ensure_ascii=False,
                indent=2,
            )

    if not args.skip_metadata:
        with (output_dir / "metadata.json").open("w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

    report_name = "conversion_report.txt"
    part_list = parse_parts(args.parts)
    if args.skip_metadata and len(part_list) == 1:
        report_name = f"conversion_report_part{part_list[0]:03d}.txt"
    with (output_dir / report_name).open("w", encoding="utf-8") as f:
        f.write(f"source_dir={source_dir}\n")
        f.write(f"output_dir={output_dir}\n")
        for part_key, splits in report["parts"].items():
            for split, stats in splits.items():
                f.write(f"{split}_{part_key}_jsonl={stats['jsonl']}\n")
                f.write(f"{split}_{part_key}_rows={stats['jsonl_rows']}\n")
                f.write(f"{split}_{part_key}_skipped={stats['skipped']}\n")

    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
