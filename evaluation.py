"""Boundary-based Khmer segmentation evaluation with no training/test leakage.

Read independent gold references from text,segmented CSV. A pipe '|' separates
gold words. These references are NEVER passed as rules to the segmenter.
"""

from __future__ import annotations  # Enable modern type hints.

import csv  # Load user-annotated gold sentence rows.
from collections import Counter  # Aggregate boundary TP/FP/FN across sentences.
from pathlib import Path  # Find the dataset safely.

from segmenter import KhmerSegmenter, METHODS, normalize  # Reuse the shared segmentation engine.


def boundaries(words: list[str]) -> set[int]:
    """Return Unicode character offsets after each word, excluding the final offset."""
    indexes: set[int] = set()  # Distinct boundary offsets; count positions, not token strings.
    cursor = 0  # Start before the first Unicode character.
    for word in words[:-1]:  # Never count the sentence-final boundary as a prediction.
        cursor += len(word)  # Offsets use Unicode code points, like the engine.
        indexes.add(cursor)  # Mark the end of this word.
    return indexes  # A one-word sentence correctly has no internal boundaries.


def evaluate(
    engine: KhmerSegmenter,
    gold_path: str | Path,
    methods: tuple[str, ...] = ("fmm", "bmm", "bimm", "smart"),
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Return sentence-level results and MICRO-averaged overall model scores."""
    if any(method not in METHODS for method in methods):  # Reject unsupported methods.
        raise ValueError("Unknown segmentation method in evaluation list.")
    predictions: list[dict[str, object]] = []  # One output row per sentence/method.
    stats = {method: Counter() for method in methods}  # Separate TP/FP/FN and exact counts.
    with Path(gold_path).open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)  # Gold dataset columns: text, segmented.
        if not {"text", "segmented"}.issubset(reader.fieldnames or []):
            raise ValueError("Gold CSV must have text,segmented columns (no blank header line).")
        for row_index, row in enumerate(reader, 2):  # Include CSV line numbers in errors.
            text = normalize((row.get("text") or "").strip())  # Normalize reference text.
            marked = normalize((row.get("segmented") or "").strip())  # Normalize annotations.
            if not text and not marked:  # Ignore intentionally blank examples.
                continue
            gold = marked.split("|")  # Each pipe marks a human-annotated word boundary.
            if any(not word for word in gold) or "".join(gold) != text:
                raise ValueError(f"Gold row {row_index}: tokens do not reconstruct text.")
            gold_b = boundaries(gold)  # Calculate reference offsets once.
            for method in methods:  # Evaluate all algorithms on the same data.
                predicted = engine.segment(text, method=method).words  # No gold-rule injection.
                pred_b = boundaries(predicted)  # Calculate predicted boundary positions.
                tp = len(gold_b & pred_b)  # Correct boundaries (true positives).
                fp = len(pred_b - gold_b)  # Incorrectly added boundaries.
                fn = len(gold_b - pred_b)  # Gold boundaries that were missed.
                precision = tp / (tp + fp) if tp + fp else (1.0 if not gold_b else 0.0)
                recall = tp / (tp + fn) if tp + fn else 1.0  # No gold boundary means full recall.
                f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
                exact = predicted == gold  # Exact token sequence match.
                stats[method].update({"tp": tp, "fp": fp, "fn": fn, "exact": int(exact), "sentences": 1})
                predictions.append({"sentence": text, "method": method.upper(),
                                    "reference_total": len(gold), "predicted_total": len(predicted),
                                    "precision": round(precision, 4), "recall": round(recall, 4),
                                    "f1": round(f1, 4), "exact": exact,
                                    "predicted_words": " | ".join(predicted)})
    overall: list[dict[str, object]] = []  # Collect one micro-averaged row per method.
    for method, count in stats.items():  # Aggregate boundary counts across sentences first.
        tp, fp, fn = count["tp"], count["fp"], count["fn"]
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        overall.append({"algorithm": method.upper(), "precision": round(100 * precision, 2),
                        "recall": round(100 * recall, 2), "f1": round(100 * f1, 2),
                        "exact_matches": f"{count['exact']}/{count['sentences']}",
                        "sentences": count["sentences"]})
    return predictions, overall  # Do not claim the small sample is real-world accuracy.
