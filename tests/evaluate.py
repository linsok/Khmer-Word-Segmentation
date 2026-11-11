"""Evaluate Maximum Matching against annotated Khmer sentences.

Run: python tests/evaluate.py
Optional models: python -X utf8 tests/evaluate.py --include-models
"""

import argparse  # Read optional command-line flags.
import sys  # Add the project root to Python's module path.
from pathlib import Path  # Find files relative to this script.

ROOT = Path(__file__).resolve().parent.parent  # Project root contains segmenter.py.
sys.path.insert(0, str(ROOT))  # Allow direct execution from any working directory.

from evaluation import evaluate  # Import the shared metric functions.
from segmenter import DEFAULT_DICTIONARY, KhmerSegmenter, load_dictionary  # Load the same engine.

parser = argparse.ArgumentParser(description="Evaluate Khmer word-boundary accuracy.")
parser.add_argument("--include-models", action="store_true", help="Also evaluate CRF and neural models")
args = parser.parse_args()  # Read CLI options.

engine = KhmerSegmenter(load_dictionary(DEFAULT_DICTIONARY))  # Build the 100k-word trie.
methods = ("fmm", "bmm", "bimm", "smart")  # Dictionary-only baseline methods.
if args.include_models:  # Use this only after khmercut[nn] is installed.
    methods += ("crf", "neural")

rows, summary = evaluate(engine, ROOT / "tests" / "gold_sentences.csv", methods)
for item in rows:  # Show each sentence's metrics without forcing a token count.
    print(f"{item['method']:6s} | gold={item['reference_total']:2d} | pred={item['predicted_total']:2d} "
          f"| F1={item['f1']:.2%} | exact={item['exact']} | {item['predicted_words']}")
print("\n--- Micro-averaged boundary accuracy (sample only) ---")
for item in summary:
    print(f"{item['algorithm']:6s}: P={item['precision']:6.2f}% R={item['recall']:6.2f}% "
          f"F1={item['f1']:6.2f}% Exact={item['exact_matches']}")
