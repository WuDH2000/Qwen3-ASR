#!/usr/bin/env python3
"""Compute mixed error rate for code-switching ASR predictions.

Chinese and Tamil spans are scored at character level. Latin-script spans
(English/Malay in the current data) are scored at whitespace word level.
"""

import argparse
import json
import re
import unicodedata
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple


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


def is_tamil_char(char: str) -> bool:
    return 0x0B80 <= ord(char) <= 0x0BFF


def normalize_text(text: str) -> str:
    text = (text or "").strip().lower()
    text = re.sub(r"[\u200b\u200c\u200d\ufeff]", "", text)
    text = re.sub(r"<\s*Speaker\s*\d+\s*>\s*:?", " ", text, flags=re.IGNORECASE)
    text = "".join(ch for ch in text if not unicodedata.category(ch).startswith("P"))
    return re.sub(r"\s+", " ", text).strip()


def mixed_units(text: str) -> List[str]:
    units: List[str] = []
    word_chars: List[str] = []

    def flush_word() -> None:
        if word_chars:
            units.append("".join(word_chars))
            word_chars.clear()

    for char in normalize_text(text):
        if char.isspace():
            flush_word()
        elif is_cjk_char(char):
            flush_word()
            units.append(char)
        elif is_tamil_char(char):
            flush_word()
            units.append(char)
        else:
            word_chars.append(char)
    flush_word()
    return units


def count_ref_unit_types(text: str) -> Tuple[int, int, int]:
    latin_words = 0
    cjk_chars = 0
    tamil_chars = 0
    word_chars: List[str] = []

    def flush_word() -> None:
        nonlocal latin_words
        if word_chars:
            latin_words += 1
            word_chars.clear()

    for char in normalize_text(text):
        if char.isspace():
            flush_word()
        elif is_cjk_char(char):
            flush_word()
            cjk_chars += 1
        elif is_tamil_char(char):
            flush_word()
            tamil_chars += 1
        else:
            word_chars.append(char)
    flush_word()
    return latin_words, cjk_chars, tamil_chars


def edit_counts(ref: Sequence[str], hyp: Sequence[str]) -> Tuple[int, int, int, int]:
    """Return distance, substitutions, insertions, deletions."""
    n = len(ref)
    m = len(hyp)
    dp = [[(0, 0, 0, 0) for _ in range(m + 1)] for _ in range(n + 1)]
    for i in range(1, n + 1):
        d, s, ins, dele = dp[i - 1][0]
        dp[i][0] = (d + 1, s, ins, dele + 1)
    for j in range(1, m + 1):
        d, s, ins, dele = dp[0][j - 1]
        dp[0][j] = (d + 1, s, ins + 1, dele)

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ref[i - 1] == hyp[j - 1]:
                match = dp[i - 1][j - 1]
            else:
                d, s, ins, dele = dp[i - 1][j - 1]
                match = (d + 1, s + 1, ins, dele)
            d, s, ins, dele = dp[i][j - 1]
            insert = (d + 1, s, ins + 1, dele)
            d, s, ins, dele = dp[i - 1][j]
            delete = (d + 1, s, ins, dele + 1)
            dp[i][j] = min(match, insert, delete, key=lambda x: (x[0], x[1], x[2], x[3]))
    return dp[n][m]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions_jsonl", required=True)
    parser.add_argument("--output_json", default="")
    args = parser.parse_args()

    total_distance = 0
    total_subs = 0
    total_inserts = 0
    total_deletes = 0
    total_ref_units = 0
    total_hyp_units = 0
    latin_words = 0
    cjk_chars = 0
    tamil_chars = 0
    rows = 0

    for line in Path(args.predictions_jsonl).open("r", encoding="utf-8"):
        if not line.strip():
            continue
        row = json.loads(line)
        ref_text = row.get("reference", "")
        hyp_text = row.get("prediction", "")
        ref_units = mixed_units(ref_text)
        hyp_units = mixed_units(hyp_text)
        dist, subs, inserts, deletes = edit_counts(ref_units, hyp_units)
        total_distance += dist
        total_subs += subs
        total_inserts += inserts
        total_deletes += deletes
        total_ref_units += len(ref_units)
        total_hyp_units += len(hyp_units)
        lw, cc, tc = count_ref_unit_types(ref_text)
        latin_words += lw
        cjk_chars += cc
        tamil_chars += tc
        rows += 1

    summary = {
        "predictions": str(Path(args.predictions_jsonl).resolve()),
        "num_samples": rows,
        "mixed_error_rate": total_distance / total_ref_units if total_ref_units else 0.0,
        "edit_distance": total_distance,
        "substitutions": total_subs,
        "insertions": total_inserts,
        "deletions": total_deletes,
        "reference_units": total_ref_units,
        "prediction_units": total_hyp_units,
        "reference_latin_words": latin_words,
        "reference_cjk_chars": cjk_chars,
        "reference_tamil_chars": tamil_chars,
        "unit_definition": "Chinese/Tamil characters + English/Malay whitespace words",
        "normalization": "lowercase, remove Unicode punctuation, remove <SpeakerN>: tags, collapse whitespace",
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if args.output_json:
        Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output_json).write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
