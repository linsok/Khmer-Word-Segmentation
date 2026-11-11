"""Compare pretrained CRF and ONNX neural models with the independent reference CSV.

Run on Windows: python -X utf8 tests/compare_models.py
Requires: python -m pip install "khmercut[nn]"
"""

from __future__ import annotations  # Support modern Python annotations.
import sys  # Make the project root importable from tests/.
from pathlib import Path  # Find the model and gold-data paths safely.

ROOT = Path(__file__).resolve().parent.parent  # Locate this project's root.
sys.path.insert(0, str(ROOT))  # Make segmenter.py visible.

from evaluation import evaluate  # Calculate micro precision, recall, and F1.
from segmenter import DEFAULT_DICTIONARY, KhmerSegmenter, load_dictionary  # Build the shared engine.


def main() -> None:
    """Evaluate both pretrained models with no preferred-word overrides."""
    engine = KhmerSegmenter(load_dictionary(DEFAULT_DICTIONARY))  # Build once.
    gold = ROOT / "tests" / "gold_sentences.csv"  # The five reference examples.
    try:  # Model dependencies are optional in the minimal installation.
        rows, totals = evaluate(engine, gold, methods=("crf", "neural"))
    except (ImportError, RuntimeError) as exc:  # Show an actionable error.
        raise SystemExit("Install optional models with: python -m pip install 'khmercut[nn]'\n" + str(exc)) from exc
    for row in rows:  # Display actual predicted tokens and sentence-level F1.
        print(f"{row['method']:6} | F1={row['f1']:.2%} | exact={row['exact']} | {row['predicted_words']}")
    print("\n--- Pretrained model summary (five demonstration sentences only) ---")
    for score in totals:  # Display overall boundary F1, not a fabricated word count.
        print(f"{score['algorithm']}: P={score['precision']}% R={score['recall']}% "
              f"F1={score['f1']}% Exact={score['exact_matches']}")


if __name__ == "__main__":  # Run only when invoked directly.
    main()
