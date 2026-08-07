#!/usr/bin/env python3
"""Run MERaLiON-3-ASR inference on a JSONL manifest and compute WER/CER."""

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
    return re.sub(r"\s+", " ", text).strip()


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
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--test_jsonl", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--backend", choices=["transformers", "vllm"], default="transformers")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--lowercase", action="store_true")
    parser.add_argument("--remove_punct", action="store_true")
    parser.add_argument("--remove_unicode_punct", action="store_true")
    parser.add_argument("--cjk_char_space", action="store_true")
    parser.add_argument("--remove_speaker_tags", action="store_true")
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

    rows = load_manifest(Path(args.test_jsonl), args.limit)
    manifest_stats = validate_manifest(rows)
    if args.dry_run:
        print(json.dumps({"test_jsonl": os.path.abspath(args.test_jsonl), **manifest_stats}, ensure_ascii=False, indent=2))
        if manifest_stats["missing_audio"] or manifest_stats["empty_reference"]:
            raise SystemExit(1)
        return
    if manifest_stats["missing_audio"] or manifest_stats["empty_reference"]:
        raise ValueError(f"Invalid manifest: {manifest_stats}")

    from meralion_3_asr import Meralion3ASR

    model = Meralion3ASR.from_pretrained(args.model_path, backend=args.backend)

    pred_path = output_dir / "predictions.jsonl"
    refs: List[str] = []
    hyps: List[str] = []
    with pred_path.open("w", encoding="utf-8") as out_f:
        for start in range(0, len(rows), args.batch_size):
            batch = rows[start : start + args.batch_size]
            predictions = model.transcribe_batch([r["audio"] for r in batch])
            for row, pred in zip(batch, predictions):
                ref = normalize_text(
                    row["reference"],
                    lowercase=args.lowercase,
                    remove_punct=args.remove_punct,
                    remove_unicode_punct=args.remove_unicode_punct,
                    cjk_char_space=args.cjk_char_space,
                    remove_speaker_tags=args.remove_speaker_tags,
                )
                hyp = normalize_text(
                    pred,
                    lowercase=args.lowercase,
                    remove_punct=args.remove_punct,
                    remove_unicode_punct=args.remove_unicode_punct,
                    cjk_char_space=args.cjk_char_space,
                    remove_speaker_tags=args.remove_speaker_tags,
                )
                refs.append(ref)
                hyps.append(hyp)
                out_f.write(json.dumps({
                    "id": row["id"],
                    "audio": row["audio"],
                    "reference": ref,
                    "prediction": hyp,
                    "sample_wer": wer(ref, hyp),
                    "sample_cer": cer(ref, hyp),
                }, ensure_ascii=False) + "\n")
            out_f.flush()
            print(f"Processed {min(start + args.batch_size, len(rows))}/{len(rows)}", flush=True)

    summary = {
        "model_path": os.path.abspath(args.model_path),
        "test_jsonl": os.path.abspath(args.test_jsonl),
        "predictions": os.path.abspath(pred_path),
        "backend": args.backend,
        **compute_metrics(refs, hyps, norm_desc),
    }
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
