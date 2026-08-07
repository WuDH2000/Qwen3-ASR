#!/usr/bin/env python3
# coding=utf-8
"""Run Qwen3-ASR inference on a JSONL manifest and compute WER/CER."""

import argparse
import json
import os
import re
import string
import unicodedata
from pathlib import Path
from typing import Dict, List, Tuple

from jiwer import cer, wer


def is_cjk_char(char: str) -> bool:
    code = ord(char)
    return (
        0x3400 <= code <= 0x4DBF
        or 0x4E00 <= code <= 0x9FFF
        or 0xF900 <= code <= 0xFAFF
        or 0x20000 <= code <= 0x2A6DF
        or 0x2A700 <= code <= 0x2B73F
        or 0x2B740 <= code <= 0x2B81F
        or 0x2B820 <= code <= 0x2CEAF
        or 0x30000 <= code <= 0x3134F
    )


def normalize_text(
    text: str,
    lowercase: bool = False,
    remove_punct: bool = False,
    remove_unicode_punct: bool = False,
    cjk_char_space: bool = False,
    remove_speaker_tags: bool = False,
) -> str:
    text = (text or "").strip()
    text = re.sub(r"[\u200b\u200c\u200d\ufeff]", "", text)
    if remove_speaker_tags:
        text = re.sub(r"<\s*Speaker\s*\d+\s*>\s*:?", " ", text, flags=re.IGNORECASE)
    if lowercase:
        text = text.lower()
    if remove_punct:
        text = text.translate(str.maketrans("", "", string.punctuation))
    if remove_unicode_punct:
        text = "".join(ch for ch in text if not unicodedata.category(ch).startswith("P"))
    if cjk_char_space:
        text = "".join(f" {ch} " if is_cjk_char(ch) else ch for ch in text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_manifest(path: Path, limit: int = 0) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            audio = row.get("audio_path") or row.get("audio")
            ref = row.get("transcription") or row.get("text")
            if isinstance(ref, str) and "<asr_text>" in ref:
                ref = ref.split("<asr_text>", 1)[1]
            if not audio or not ref:
                raise ValueError(f"Missing audio/text field in row: {row}")
            rows.append({"audio": audio, "reference": ref, "id": str(row.get("id", len(rows)))})
            if limit and len(rows) >= limit:
                break
    return rows


def validate_manifest(rows: List[Dict[str, str]]) -> Dict[str, int]:
    missing_audio = 0
    empty_reference = 0
    for row in rows:
        if not os.path.exists(row["audio"]):
            missing_audio += 1
        if not row["reference"].strip():
            empty_reference += 1
    return {"num_samples": len(rows), "missing_audio": missing_audio, "empty_reference": empty_reference}


def load_predictions(
    path: Path,
    lowercase: bool,
    remove_punct: bool,
    remove_unicode_punct: bool,
    cjk_char_space: bool,
    remove_speaker_tags: bool,
) -> Tuple[List[str], List[str]]:
    refs: List[str] = []
    hyps: List[str] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            refs.append(
                normalize_text(
                    row.get("reference", ""),
                    lowercase=lowercase,
                    remove_punct=remove_punct,
                    remove_unicode_punct=remove_unicode_punct,
                    cjk_char_space=cjk_char_space,
                    remove_speaker_tags=remove_speaker_tags,
                )
            )
            hyps.append(
                normalize_text(
                    row.get("prediction", ""),
                    lowercase=lowercase,
                    remove_punct=remove_punct,
                    remove_unicode_punct=remove_unicode_punct,
                    cjk_char_space=cjk_char_space,
                    remove_speaker_tags=remove_speaker_tags,
                )
            )
    return refs, hyps


def compute_metrics(refs: List[str], hyps: List[str], normalization: str) -> Dict[str, object]:
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
        "normalization": normalization,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default="")
    parser.add_argument("--test_jsonl", required=False)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--language", default=None)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--lowercase", action="store_true")
    parser.add_argument("--remove_punct", action="store_true")
    parser.add_argument("--remove_unicode_punct", action="store_true")
    parser.add_argument("--cjk_char_space", action="store_true")
    parser.add_argument("--remove_speaker_tags", action="store_true")
    parser.add_argument("--predictions_jsonl", default="")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    norm_desc = "strip, remove zero-width chars, collapse whitespace"
    if args.lowercase:
        norm_desc += ", lowercase"
    if args.remove_punct:
        norm_desc += ", remove ASCII punctuation"
    if args.remove_unicode_punct:
        norm_desc += ", remove Unicode punctuation"
    if args.cjk_char_space:
        norm_desc += ", add spaces around CJK ideographs for WER tokenization"
    if args.remove_speaker_tags:
        norm_desc += ", remove <SpeakerN>: tags"

    if args.predictions_jsonl:
        refs, hyps = load_predictions(
            Path(args.predictions_jsonl),
            args.lowercase,
            args.remove_punct,
            args.remove_unicode_punct,
            args.cjk_char_space,
            args.remove_speaker_tags,
        )
        summary = {"predictions": os.path.abspath(args.predictions_jsonl), **compute_metrics(refs, hyps, norm_desc)}
        with (output_dir / "metrics.json").open("w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return

    if not args.test_jsonl:
        raise ValueError("--test_jsonl is required unless --predictions_jsonl is used")
    rows = load_manifest(Path(args.test_jsonl), args.limit)
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
                language=args.language,
            )
            for row, result in zip(batch, results):
                ref = normalize_text(
                    row["reference"],
                    lowercase=args.lowercase,
                    remove_punct=args.remove_punct,
                    remove_unicode_punct=args.remove_unicode_punct,
                    cjk_char_space=args.cjk_char_space,
                    remove_speaker_tags=args.remove_speaker_tags,
                )
                hyp = normalize_text(
                    result.text,
                    lowercase=args.lowercase,
                    remove_punct=args.remove_punct,
                    remove_unicode_punct=args.remove_unicode_punct,
                    cjk_char_space=args.cjk_char_space,
                    remove_speaker_tags=args.remove_speaker_tags,
                )
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
                    "forced_language": args.language,
                }
                out_f.write(json.dumps(out, ensure_ascii=False) + "\n")
            print(f"Processed {min(start + args.batch_size, len(rows))}/{len(rows)}", flush=True)

    summary = {
        "model_path": os.path.abspath(args.model_path),
        "test_jsonl": os.path.abspath(args.test_jsonl),
        "predictions": str(pred_path),
        "forced_language": args.language,
        **compute_metrics(refs, hyps, norm_desc),
    }
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
