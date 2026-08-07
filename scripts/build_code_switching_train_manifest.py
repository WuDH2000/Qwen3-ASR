#!/usr/bin/env python3
# coding=utf-8
"""Build a mixed Tamil + Singlish/code-switching Qwen3-ASR training manifest."""

import argparse
import json
import random
from pathlib import Path
from typing import Dict, Iterable, List


DEFAULT_TAMIL = "/mnt/weka/aisg/speech_spoke/donghang/data/deepdml-iisc-mile-tamil-asr/train.jsonl"
DEFAULT_MERALION = "/mnt/weka/aisg/speech_spoke/donghang/data/meralion-multitask-national-speech-corpus-v1"
DEFAULT_OUTPUT = (
    "/mnt/weka/aisg/speech_spoke/donghang/data/code-switching/"
    "train_tamil_part1_3_45000_part4.jsonl"
)


def iter_jsonl(path: Path) -> Iterable[Dict[str, object]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def normalize_row(row: Dict[str, object], source: str) -> Dict[str, object]:
    audio = row.get("audio") or row.get("audio_path")
    text = row.get("text")
    if not audio or not text:
        raise ValueError(f"Missing audio/text in {source}: {row}")
    out = dict(row)
    out["audio"] = audio
    out["audio_path"] = row.get("audio_path") or audio
    out["text"] = text
    out["source_manifest"] = source
    out.setdefault("prompt", row.get("source_instruction", ""))
    return out


def reservoir_sample(path: Path, k: int, rng: random.Random, source: str) -> List[Dict[str, object]]:
    sample: List[Dict[str, object]] = []
    seen = 0
    for row in iter_jsonl(path):
        item = normalize_row(row, source)
        seen += 1
        if len(sample) < k:
            sample.append(item)
            continue
        j = rng.randrange(seen)
        if j < k:
            sample[j] = item
    if seen < k:
        raise ValueError(f"Requested {k} rows from {path}, but only found {seen}")
    return sample


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tamil-train", default=DEFAULT_TAMIL)
    parser.add_argument("--meralion-dir", default=DEFAULT_MERALION)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--samples-per-english-part", type=int, default=15000)
    parser.add_argument("--seed", type=int, default=20260802)
    args = parser.parse_args()

    tamil_path = Path(args.tamil_train)
    meralion_dir = Path(args.meralion_dir)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)
    rows: List[Dict[str, object]] = []
    counts: Dict[str, int] = {}

    for part in (1, 2, 3):
        path = meralion_dir / f"train_part{part:03d}.jsonl"
        source = f"meralion_part{part:03d}_sampled"
        picked = reservoir_sample(path, args.samples_per_english_part, rng, source)
        rows.extend(picked)
        counts[source] = len(picked)

    part4_path = meralion_dir / "train_part004.jsonl"
    part4_rows = [normalize_row(row, "meralion_part004_all") for row in iter_jsonl(part4_path)]
    rows.extend(part4_rows)
    counts["meralion_part004_all"] = len(part4_rows)

    tamil_rows = [normalize_row(row, "deepdml_tamil_all") for row in iter_jsonl(tamil_path)]
    rows.extend(tamil_rows)
    counts["deepdml_tamil_all"] = len(tamil_rows)

    rng.shuffle(rows)
    with output_path.open("w", encoding="utf-8") as out_f:
        for idx, row in enumerate(rows):
            row["mix_id"] = f"codeswitch_mix_{idx:06d}"
            out_f.write(json.dumps(row, ensure_ascii=False) + "\n")

    report = {
        "output": str(output_path),
        "seed": args.seed,
        "samples_per_english_part": args.samples_per_english_part,
        "counts": counts,
        "total_rows": len(rows),
    }
    report_path = output_path.with_suffix(".metadata.json")
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
