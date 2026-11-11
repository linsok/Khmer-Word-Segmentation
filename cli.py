"""Command-line Khmer segmentation and CSV batch processing.

Examples:
    python -X utf8 cli.py --text "ខ្ញុំស្រឡាញ់ភាសាខ្មែរ" --method fmm
    python -X utf8 cli.py --input-csv examples/input_sentences.csv --text-column text
"""

from __future__ import annotations  # Enable Python 3.10+ annotation syntax.

import argparse  # Parse terminal options and document their usage.
from pathlib import Path  # Work with Windows/Linux paths safely.

from csv_tools import batch_segment, load_csv, output_csv  # Reuse the app's exact CSV logic.
from segmenter import DEFAULT_DICTIONARY, METHODS, KhmerSegmenter, load_dictionary  # Reuse one engine.


def main() -> None:
    """Process a single input or an entire CSV file with the same NLP settings."""
    parser = argparse.ArgumentParser(description="Khmer word segmentation and vocabulary extraction")
    parser.add_argument("--text", default="ខ្ញុំស្រឡាញ់ភាសាខ្មែរ", help="Input Khmer sentence")
    parser.add_argument("--interactive", action="store_true", help="Prompt for Khmer text")
    parser.add_argument("--method", choices=METHODS, default="smart", help="FMM, BMM, BiMM, SMART, CRF or neural")
    parser.add_argument("--dictionary", type=Path, default=DEFAULT_DICTIONARY, help="Dictionary with word,pos columns")
    parser.add_argument("--input-csv", type=Path, help="Batch-process arbitrary CSV input")
    parser.add_argument("--text-column", help="CSV column containing Khmer text")
    parser.add_argument("--delimiter", default="auto", choices=("auto", ",", ";", "tab", "|"))
    parser.add_argument("--output", type=Path, default=Path("segmented_output.csv"), help="Processed batch CSV destination")
    parser.add_argument("--vocab-output", type=Path, default=Path("unique_vocabulary.csv"), help="Unique vocabulary CSV destination")
    parser.add_argument("--keep-stopwords", action="store_true", help="Keep stopwords in cleaned vocabulary")
    parser.add_argument("--exclude-numbers", action="store_true", help="Exclude numbers from vocabulary")
    args = parser.parse_args()  # Read the user-selected command-line arguments.
    engine = KhmerSegmenter(load_dictionary(args.dictionary))  # Build dictionary trie once.
    if args.input_csv:  # Process a CSV instead of the single-sentence mode.
        table = load_csv(args.input_csv.read_bytes(), delimiter=args.delimiter)  # Decode UTF-8/UTF-16.
        if not args.text_column:  # Any header is accepted, but the user must identify the text field.
            parser.error(f"For CSV processing, add --text-column. Available: {table.fields}")
        fields, rows, vocabulary = batch_segment(  # Preserve all original input columns.
            table, args.text_column, engine, method=args.method,
            remove_stopwords=not args.keep_stopwords,
            include_numbers=not args.exclude_numbers,
        )
        args.output.write_bytes(output_csv(fields, rows))  # Excel-friendly UTF-8 BOM export.
        args.vocab_output.write_bytes(output_csv(["token", "frequency"], vocabulary))
        print(f"Processed {len(rows)} CSV rows, {len(vocabulary)} unique vocabulary tokens.")
        print(f"Rows: {args.output.resolve()}")  # Show the exact output location.
        print(f"Vocabulary: {args.vocab_output.resolve()}")
        return  # Skip the single-sentence mode when a CSV was supplied.
    sentence = input("Enter Khmer text: ") if args.interactive else args.text  # Read user input.
    result = engine.segment(sentence, args.method)  # Segment without manual word preferences.
    result.show(detail=True)  # Print screenshot-style fields plus optional statistics.
    content = result.vocabulary(remove_stopwords=not args.keep_stopwords,
                                include_numbers=not args.exclude_numbers)
    print("Vocabulary (deduplicated):", [row["token"] for row in content])  # Separate from the sentence.


if __name__ == "__main__":  # Run main only when invoked from the terminal.
    main()  # Execute the selected segmentation workflow.
