#!/usr/bin/env python3
# coding=utf-8
"""Run Qwen3-ASR inference on FLEURS Tamil and compute WER/CER."""

import argparse
import json
import os
import re
from pathlib import Path
from typing import Dict, List, Tuple

from jiwer import cer, wer


DEFAULT_TEST = "/mnt/weka/aisg/speech_spoke/donghang/data/fleur/tamil/test.jsonl"
DEFAULT_OUT = "/mnt/weka/aisg/speech_spoke/donghang/project/Qwen3-ASR/outputs/fleurs_tamil_eval"


def normalize_tamil_text(text: str) -> str:
    text = (text or "").strip()
    text = re.sub(r"[\u200b\u200c\u200d\ufeff]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text


def load_manifest(path: Path, limit: int = 0) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            audio = row.get("audio_path") or row.get("audio")
            ref = row.get("text") or row.get("transcription")
            if not audio or not ref:
                raise ValueError(f"Missing audio/text field in row: {row}")
            rows.append({"audio": audio, "reference": ref, "id": row.get("id", str(len(rows)))})
            if limit and len(rows) >= limit:
                break
    return rows


def validate_manifest(rows: List[Dict[str, str]]) -> Dict[str, int]:
    missing_audio = 0
    empty_reference = 0
    for row in rows:
        if not os.path.exists(row["audio"]):
            missing_audio += 1
        if not normalize_tamil_text(row["reference"]):
            empty_reference += 1
    return {
        "num_samples": len(rows),
        "missing_audio": missing_audio,
        "empty_reference": empty_reference,
    }


def load_predictions(path: Path) -> Tuple[List[str], List[str]]:
    refs: List[str] = []
    hyps: List[str] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            refs.append(normalize_tamil_text(row.get("reference", "")))
            hyps.append(normalize_tamil_text(row.get("prediction", "")))
    return refs, hyps


def compute_metrics(refs: List[str], hyps: List[str]) -> Dict[str, object]:
    if len(refs) != len(hyps):
        raise ValueError(f"Reference/prediction size mismatch: {len(refs)} vs {len(hyps)}")
    return {
        "num_samples": len(refs),
        "wer": wer(refs, hyps),
        "cer": cer(refs, hyps),
        "reference_words": sum(len(x.split()) for x in refs),
        "prediction_words": sum(len(x.split()) for x in hyps),
        "reference_chars_no_space": sum(len(x.replace(" ", "")) for x in refs),
        "prediction_chars_no_space": sum(len(x.replace(" ", "")) for x in hyps),
        "normalization": "strip, remove zero-width chars, collapse whitespace",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default="")
    parser.add_argument("--test_jsonl", default=DEFAULT_TEST)
    parser.add_argument("--output_dir", default=DEFAULT_OUT)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--predictions_jsonl", default="", help="Recompute metrics from an existing predictions file.")
    parser.add_argument("--dry_run", action="store_true", help="Validate manifest/audio paths without loading a model.")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.predictions_jsonl:
        refs, hyps = load_predictions(Path(args.predictions_jsonl))
        summary = {
            "predictions": os.path.abspath(args.predictions_jsonl),
            **compute_metrics(refs, hyps),
        }
        with (output_dir / "metrics.json").open("w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return

    rows = load_manifest(Path(args.test_jsonl), limit=args.limit)
    manifest_stats = validate_manifest(rows)
    if args.dry_run:
        print(json.dumps({"test_jsonl": os.path.abspath(args.test_jsonl), **manifest_stats}, ensure_ascii=False, indent=2))
        if manifest_stats["missing_audio"] or manifest_stats["empty_reference"]:
            raise SystemExit(1)
        return

    if not args.model_path:
        raise ValueError("--model_path is required unless --dry_run or --predictions_jsonl is used")
    if manifest_stats["missing_audio"] or manifest_stats["empty_reference"]:
        raise ValueError(f"Invalid manifest: {manifest_stats}")

    import torch
    from qwen_asr import Qwen3ASRModel

    use_bf16 = torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8
    model = Qwen3ASRModel.from_pretrained(
        args.model_path,
        dtype=torch.bfloat16 if use_bf16 else (torch.float16 if torch.cuda.is_available() else torch.float32),
        device_map="cuda:0" if torch.cuda.is_available() else None,
        max_inference_batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
    )

    pred_path = output_dir / "predictions.jsonl"
    refs: List[str] = []
    hyps: List[str] = []
    with pred_path.open("w", encoding="utf-8") as out_f:
        for start in range(0, len(rows), args.batch_size):
            batch = rows[start : start + args.batch_size]
            results = model.transcribe(
                audio=[r["audio"] for r in batch],
                language=None,
            )
            for row, result in zip(batch, results):
                ref = normalize_tamil_text(row["reference"])
                hyp = normalize_tamil_text(result.text)
                refs.append(ref)
                hyps.append(hyp)
                out = {
                    "id": row["id"],
                    "audio": row["audio"],
                    "reference": ref,
                    "prediction": hyp,
                    "sample_wer": wer(ref, hyp),
                    "sample_cer": cer(ref, hyp),
                    "detected_language": result.language,
                }
                out_f.write(json.dumps(out, ensure_ascii=False) + "\n")
            print(f"Processed {min(start + args.batch_size, len(rows))}/{len(rows)}", flush=True)

    summary = {
        "model_path": os.path.abspath(args.model_path),
        "test_jsonl": os.path.abspath(args.test_jsonl),
        "predictions": str(pred_path),
        **compute_metrics(refs, hyps),
    }
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
