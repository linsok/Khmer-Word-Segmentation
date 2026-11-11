# Quality assurance and known limitations

## What was corrected from the uploaded ZIP

- `app.py`, the notebook, `README.md`, `tests/reference_sentences.json`, and `tests/test_segmenter.py` were empty in the uploaded project. They now contain a working implementation, documented examples and automated checks.
- Organized shared source modules: `segmenter.py`, `csv_tools.py`, and `evaluation.py` used consistently by the app, notebook, tests and CLI.
- Preserved the original 101,134-row dictionary byte-for-byte. It contains 101,134 unique words; 65,387 entries have blank POS. There are 95 entries containing whitespace and 2 punctuation-only entries, leaving 101,037 entries suitable for ordinary matching.
- Created FMM, BMM and BiMM over Unicode grapheme clusters using the full maximum dictionary entry length (22 clusters, no default manual cap).
- Restored a general SMART dynamic-programming scoring baseline without sentence-specific preferred-word lists. Added optional external CRF and neural segmentation via `khmercut[nn]`.
- Preserved text across Unicode normalization, explicit existing `U+200B` boundaries, punctuation, digits, mixed Latin text and symbols.
- Kept total word occurrences separate from unique vocabulary. Removed duplicates only from stopword sets and vocabulary rows. Optional cleaning removes stopwords, Latin casing and numbers from vocabulary, not from the original text.
- Added selectable-column CSV import/export (UTF-8, UTF-8 with BOM, UTF-16), row-preserving output and a separate corpus vocabulary file.

## Checks performed

- `python -m py_compile`: all Python modules compiled successfully.
- `python -m pytest -q`: **49 tests passed**.
- `python tests/evaluate.py`: FMM/BMM/BiMM and SMART produced documented five-sentence sample results.
- All **14 Jupyter notebook code cells** executed successfully (30 total cells including explanatory Markdown) with no exceptions.
- CSV command-line processing completed **5 demonstration rows**, preserving source columns and producing two readable CSV exports.
- ZIP integrity test and file completeness checks are performed before delivery.

## Accuracy caveats

- Word segmentation is linguistically ambiguous. The supplied 11-word segmentation of `ទឹកត្រីដែលឆ្ងាញ់គឺធ្វើមកពីត្រីដែលរស់នៅក្នុងទឹកសាប` is recorded as a **reference annotation**, not a rule injected into the algorithms.
- On that sentence the offline FMM baseline returns 10 tokens and SMART returns 14 tokens. They preserve the text but do not exactly match all reference boundaries.
- Aggregate sample boundary-F1 on only five demonstration sentences is 89.47% for FMM/BMM/BiMM and 93.02% for SMART. These are **not** claims of accuracy on arbitrary Khmer input.
- The optional pretrained CRF/neural models may give different compound-word boundaries; their quality should be evaluated on new unseen annotations.
- Live Streamlit browser rendering was not verified in this build environment because the `streamlit` package was not available and network package installation was blocked. The module passes Python compilation, but the web UI should be visually checked after installing dependencies on your computer.

## Suggested next step for research quality

Build a larger gold corpus with explicit, consistent word-boundary annotation guidelines; split into training/dev/test by document, then measure boundary precision, recall and F1 on **unseen** documents. Only train or calibrate segmentation weights on the training/dev partitions; never tune against the final test set.
