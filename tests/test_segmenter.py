"""Regression tests for Unicode, word boundaries, vocabulary, CSV and evaluation.

Run from project root: python -m pytest -q
The tests deliberately do not force an 11-word example by special preferences.
"""

from __future__ import annotations  # Use modern Python type annotations.

import sys  # Ensure tests can import project modules from any working directory.
from pathlib import Path  # Build platform-independent project paths.

import pytest  # Use pytest fixtures and parametrization.

ROOT = Path(__file__).resolve().parent.parent  # Resolve the containing project root.
sys.path.insert(0, str(ROOT))  # Import source files without installing a wheel.

from csv_tools import batch_segment, load_csv, output_csv  # Verify CSV workflows.
from evaluation import boundaries, evaluate  # Verify gold-standard F1 calculation.
from segmenter import (  # Verify the shared segmentation API.
    DEFAULT_DICTIONARY, DEFAULT_STOPWORDS, ZERO_WIDTH_SPACE,
    KhmerSegmenter, clusters, dictionary_report, load_dictionary, normalize,
)


@pytest.fixture(scope="module")
def engine() -> KhmerSegmenter:
    """Build the large dictionary trie once for this test module."""
    return KhmerSegmenter(load_dictionary(DEFAULT_DICTIONARY))  # All methods share this trie.


def test_original_dictionary_has_expected_unique_entries() -> None:
    """Confirm the supplied 101,134-row source was not replaced by a tiny sample."""
    report = dictionary_report(DEFAULT_DICTIONARY)  # Inspect raw source CSV.
    assert report["rows"] == 101134  # Original rows from the user's dictionary.
    assert report["unique_words"] == 101134  # The source has no duplicate words.
    assert report["usable_words"] == 101037  # Filter 95 spaced + 2 punctuation entries.
    assert report["missing_pos"] == 65387  # Empty POS labels are preserved as blanks.


def test_grapheme_cluster_keeps_khmer_marks() -> None:
    """Khmer orthographic clusters should not detach vowel/Coeng marks."""
    assert clusters("ខ្ញុំ") == ["ខ្ញុំ"]  # A grapheme cluster can contain several code points.
    assert "".join(clusters("សាលា")) == "សាលា"  # Unicode input must round-trip.


def test_fmm_picks_longest_valid_word(engine: KhmerSegmenter) -> None:
    """FMM must recognize ទឹកត្រី as one dictionary word, not two."""
    prediction = engine.segment("ទឹកត្រីដែលឆ្ងាញ់", "fmm")
    assert prediction.words[:3] == ["ទឹកត្រី", "ដែល", "ឆ្ងាញ់"]


def test_no_hardcoded_11_word_claim(engine: KhmerSegmenter) -> None:
    """Actual algorithms may disagree with gold; they must report their real output."""
    sample = "ទឹកត្រីដែលឆ្ងាញ់គឺធ្វើមកពីត្រីដែលរស់នៅក្នុងទឹកសាប"
    result = engine.segment(sample, method="fmm")
    assert result.total_words == len(result.words)  # Count tokens, not an expected constant.
    assert result.total_words == 10  # Baseline longest matching on this dictionary.
    assert result.reconstructed == result.normalized  # No Unicode characters discarded.


def test_existing_zwsp_is_hard_boundary(engine: KhmerSegmenter) -> None:
    """Never match a dictionary word across explicit invisible separators."""
    source = "ខ្ញុំ\u200bស្រឡាញ់\u200bភាសាខ្មែរ"
    result = engine.segment(source, method="smart")
    assert result.words[:2] == ["ខ្ញុំ", "ស្រឡាញ់"]  # Existing boundaries remain.
    assert result.reconstructed == source  # Input ZWSP is retained in lossless output.
    assert result.zero_space.count(ZERO_WIDTH_SPACE) == result.total_words - 1


@pytest.mark.parametrize("text", [
    "", "ខ្ញុំស្រឡាញ់ភាសាខ្មែរ", "  សួស្តីបងប្អូន!  ",
    "ខ្ញុំ\u200bកំពុង\u200bរៀន", "ថ្ងៃនេះខ្ញុំរៀន Python 3.14។",
    "សិស្សអានសៀវភៅ។\nគាត់ទៅសាលារៀន។",
    "១២,៥០០៛", "ខ្ញុំ\t\u00a0សុខសប្បាយ", "🧡សួស្តី👋",
])
@pytest.mark.parametrize("method", ["fmm", "bmm", "bimm", "smart"])
def test_input_is_lossless(engine: KhmerSegmenter, text: str, method: str) -> None:
    """Spaces, punctuation, digits and emoji must survive every dictionary method."""
    prediction = engine.segment(text, method=method)
    assert prediction.reconstructed == normalize(text)
    assert "".join(part.text for part in prediction.parts) == normalize(text)
    assert all(normalize(text)[p.start:p.end] == p.text for p in prediction.parts)


def test_stopwords_are_deduplicated_and_keep_sentence(engine: KhmerSegmenter) -> None:
    """Stopwords are removed only in content vocabulary, never from raw segmentation."""
    own = KhmerSegmenter(engine.vocabulary, stopwords=["ដែល", "ដែល", "គឺ"])
    result = own.segment("ដែល\u200bដែល\u200bគឺ\u200bត្រី", method="fmm")
    assert result.words == ["ដែល", "ដែល", "គឺ", "ត្រី"]  # Raw words contain repetition.
    assert len(own.stopwords) == 2  # Duplicate stopwords collapse to one set entry.
    assert result.cleaned_tokens(remove_stopwords=True) == ["ត្រី"]
    assert result.vocabulary(remove_stopwords=True) == [{"token": "ត្រី", "frequency": 1}]
    assert result.total_words == 4  # Source total must not be replaced by vocabulary size.


def test_latin_casefold_and_numbers(engine: KhmerSegmenter) -> None:
    """Latin lowercase, punctuation omission and number options affect vocabulary only."""
    result = engine.segment("Python Python 12.5។", method="smart")
    assert result.words == ["Python", "Python", "12.5"]
    assert result.cleaned_tokens(remove_stopwords=True, include_numbers=False) == ["python", "python"]
    assert result.vocabulary(remove_stopwords=True, include_numbers=False) == [{"token": "python", "frequency": 2}]
    assert result.reconstructed == "Python Python 12.5។"


def test_number_with_khmer_digits(engine: KhmerSegmenter) -> None:
    """Khmer numerals should remain one token without treating punctuation as words."""
    result = engine.segment("១២,៥០០៛។", method="smart")
    assert result.words == ["១២,៥០០"]
    assert result.reconstructed == "១២,៥០០៛។"


def test_csv_import_any_column_and_row_preservation(engine: KhmerSegmenter) -> None:
    """Custom Khmer text column names are supported, including empty rows."""
    data = ("code;khmer_sentence;note\n"
            "a;ខ្ញុំស្រឡាញ់ភាសាខ្មែរ;first\n"
            "b;;empty\n").encode("utf-8-sig")
    table = load_csv(data, delimiter="auto")
    assert table.fields == ["code", "khmer_sentence", "note"]
    assert len(table.rows) == 2
    fields, rows, corpus = batch_segment(table, "khmer_sentence", engine, method="smart")
    assert len(rows) == 2  # Do not skip empty user rows.
    assert rows[0]["code"] == "a" and rows[1]["note"] == "empty"
    assert rows[1]["seg_total_words"] == 0  # Blank text is valid.
    assert len(corpus) == len({entry["token"] for entry in corpus})  # Unique vocabulary.
    exported = output_csv(fields, rows)
    assert exported.startswith(b"\xef\xbb\xbf")  # Excel-friendly UTF-8 BOM.
    reopened = load_csv(exported)  # Download can be re-imported.
    assert len(reopened.rows) == len(rows)
    assert reopened.rows[0]["khmer_sentence"] == table.rows[0]["khmer_sentence"]


def test_utf16_csv_import() -> None:
    """Users with UTF-16 CSV exports should not lose Khmer text."""
    source = "sentence\nសួស្តីបងប្អូន\n".encode("utf-16")
    table = load_csv(source)
    assert table.rows[0]["sentence"] == "សួស្តីបងប្អូន"


def test_csv_malformed_row_raises() -> None:
    """Do not silently discard unquoted commas or other malformed CSV content."""
    with pytest.raises(ValueError):
        load_csv(b"a,b\n1,2,3\n")


def test_evaluation_exposes_real_gold_mismatches(engine: KhmerSegmenter) -> None:
    """Golden labels evaluate accuracy; they are not applied as segmentation rules."""
    rows, scores = evaluate(engine, ROOT / "tests" / "gold_sentences.csv", methods=("fmm", "smart"))
    assert len(rows) == 10  # Five reference sentences × two algorithms.
    assert len(scores) == 2
    assert scores[0]["exact_matches"] == "3/5"  # Known FMM baseline on five examples.
    assert rows[0]["exact"] is False  # No hidden configuration to force 11 words.


def test_single_word_has_no_internal_boundary() -> None:
    """Word boundaries are positions between tokens, not words themselves."""
    assert boundaries(["ទឹកត្រី"]) == set()
    assert boundaries(["ទឹកត្រី", "ដែល"]) == {len("ទឹកត្រី")}
