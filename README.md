# Khmer Word Segmentation Studio 🇰🇭

**A complete educational Khmer NLP project** restored and corrected from the supplied `KH-segmentation.zip`. The user's original 101,134-row dictionary is included unchanged (SHA-256: `52561280e9b7c1c1558c79fdd350b38ab6cbcff83144eb392bc5984aaa783c15`). This implementation uses the research linked below as inspiration; it is independently written.

## 1. Run on Windows / VS Code

Open a terminal inside the extracted **`KH-segmentation`** folder (the folder containing `app.py` and `segmenter.py`). Verify Python 3.10+:

```powershell
python --version
```

Install the packages for the graphical app, Jupyter and optional pretrained segmentation:

```powershell
python -m pip install -r requirements.txt
```

Start the Streamlit application using UTF-8 mode (important on Windows for the pretrained neural model):

```powershell
python -X utf8 -m streamlit run app.py
```

Open the local address shown in the terminal, usually `http://localhost:8501`. You can also double-click `run_app.bat`.

**If you only need FMM/BMM/BiMM/SMART and the command-line interface**, use the smaller dependency set:

```powershell
python -m pip install -r requirements-minimal.txt
python -X utf8 segmenter.py --interactive --method fmm
```

**If Streamlit isn't found:** rerun installation with the *same* `python` executable as the one used to launch the app. The app is not a web-hosted service; it runs locally on your laptop.

## 2. Jupyter Notebook

```powershell
python -X utf8 -m notebook
```

From the Jupyter browser choose `notebooks/khmer_segmentation.ipynb`, then **Run All**. The notebook imports the same `segmenter.py` as the app and comments each important operation. If desired, regenerate it with `python make_notebook.py`.

## 3. Files in this project

| File / folder | Role |
| --- | --- |
| `app.py` | Responsive Streamlit interface: input sentence, algorithm choice, colored tokens, CSV import/export, unique vocabulary and evaluation |
| `segmenter.py` | Shared UTF-8/Unicode segmentation engine, FMM/BMM/BiMM, SMART, optional CRF/neural, POS and stopword analysis |
| `csv_tools.py` | Imports CSV files with selectable headers and delimiters; processes every row; exports UTF-8 BOM CSV |
| `evaluation.py` | Computes character-boundary precision, recall, F1 and exact matches |
| `cli.py` | Terminal mode and CSV batch processing |
| `datasets/cleaned_khmer_dictionary.csv` | Original 101,134-row Khmer dictionary, unmodified |
| `notebooks/khmer_segmentation.ipynb` | Commented interactive notebook and screenshot-like `sentence.show()` output |
| `tests/gold_sentences.csv` | Five sample reference annotations for *evaluation only* |
| `tests/test_segmenter.py` | Automated regression tests |
| `tests/compare_models.py` | CRF vs neural scoring (requires pretrained package) |
| `examples/input_sentences.csv` | Example CSV with a `khmer_sentence` text column |
| `requirements.txt` | Full app, notebook, and model dependencies |
| `requirements-minimal.txt` | Offline baseline dependencies without pretrained models |

## 4. Difference between segmentation and unique vocabulary

The pipeline is **input → NFC Unicode normalization → identify existing zero-width spaces, punctuation, numbers, and Khmer runs → segment the Khmer runs → calculate word occurrences → optional stopword removal and Latin casefolding → unique vocabulary**.

- **Segmented words** preserve *every* word, even repeated words. Removing duplicates here would corrupt the meaning of the sentence.
- **Total Words** counts all occurrences in that segmentation.
- **Unique Tokens** counts distinct word spellings before stopword removal.
- **Clean Vocabulary** contains one row per token, after optional stopword filtering, with a frequency column.
- **Stopwords** are kept in a deduplicated set; by default they are not removed until the vocabulary-analysis stage.
- **Whitespace and punctuation** do not count as word tokens; the original input and its boundaries are still preserved in `Result.reconstructed` and `Result.zero_space_preserved`.
- **ZeroSpace** inserts `U+200B` between predicted words; it does not magically make the predictions correct.
- **Maximum Matching** uses a complete trie and considers the full maximum dictionary word length by default; partial dictionary candidates are compared using actual Khmer grapheme clusters, not raw code-point counts.

### Screenshot-style result

```python
from segmenter import KhmerSegmenter, load_dictionary
segmenter = KhmerSegmenter(load_dictionary())
sentence = segmenter.segment('ខ្ញុំស្រឡាញ់ភាសាខ្មែរ', method='smart')
sentence.show()
```

Output with the supplied dictionary:

```text
Original Text: ខ្ញុំស្រឡាញ់ភាសាខ្មែរ
Segment: ['ខ្ញុំ', 'ស្រឡាញ់', 'ភាសា', 'ខ្មែរ']
ZeroSpace: ខ្ញុំ​ស្រឡាញ់​ភាសា​ខ្មែរ
Total Words: 4
```

Note: The `ZeroSpace` line contains invisible `U+200B` characters. Models may disagree on the correct compound-word boundaries.

## 5. What about the 11-word example?

Your manually annotated example is stored in `tests/gold_sentences.csv`:

`ទឹកត្រី|ដែល|ឆ្ងាញ់|គឺ|ធ្វើ|មកពី|ត្រី|ដែល|រស់នៅ|ក្នុង|ទឹកសាប`

It has **11 word occurrences and 10 distinct tokens** (the token `ដែល` appears twice). FMM/BMM/SMART or pretrained CRF/neural may produce a different segmentation. We **do not** secretly hardcode those boundaries or apply `--prefer-word` settings to make the demonstration pass. Correct boundaries are an empirical NLP problem. Improve results with verified, consistently annotated training data, maintain separate training/dev/test data, and report independent boundary F1 and exact-match rates.

To compare the built-in algorithms with your own dictionary:

```powershell
python -X utf8 segmenter.py --interactive --compare
python -X utf8 tests/evaluate.py
```

Optional pretrained models:

```powershell
python -m pip install "khmercut[nn]"
python -X utf8 tests/compare_models.py
```

On Windows `python -X utf8` avoids a known encoding failure when the optional `khmercut` neural model reads its Unicode vocabulary file.

## 6. CSV import and export

**Inside the Streamlit app:** open **CSV Batch**, upload a CSV, select its delimiter (`auto`, comma, semicolon, tab or pipe) and pick the column containing Khmer text. Download two CSVs: (1) **Processed rows** with all original columns plus segmentation columns, and (2) **Corpus vocabulary** with unique cleaned tokens and their frequencies.

**In the terminal:**

```powershell
python -X utf8 cli.py --input-csv examples/input_sentences.csv --text-column khmer_sentence --method smart --output my_segmented_rows.csv --vocab-output my_unique_vocabulary.csv
```

The program supports well-formed UTF-8, UTF-8-BOM and UTF-16 input CSV files. It preserves original row order, field values, and empty cells in the *processed output*. It never modifies the uploaded/original CSV. Excel-friendly UTF-8 BOM output is produced. For very large inputs use the CLI rather than the interactive Streamlit batch tab.

## 7. Evaluation and tests

```powershell
python -m pytest -q
python -X utf8 tests/evaluate.py
```

The five example gold sentences form a small learning/demo set; high scores on them **do not** demonstrate general real-world accuracy. Extend `tests/gold_sentences.csv` with additional independent annotations and evaluate on *unseen* sentences. A similar word count alone does not imply matching token boundaries.

## 8. Dependencies and provenance

- Python 3.10+
- `regex` for Unicode extended grapheme matching
- `streamlit`, `pandas` for the interface and tables
- `notebook`, `ipykernel`, `nbformat` for Jupyter
- `pytest` for tests
- Optional `khmercut[nn]` for CRF and neural segmentation (included in the full requirements)

Research reference (conceptual inspiration, not copied source code): [MaoMakara/KhmerWordSegmentation](https://github.com/MaoMakara/KhmerWordSegmentation), and Makara et al., *Bi-directional Maximal Matching Algorithm to Segment Khmer Words in Sentence*, Journal of Information Processing Systems (2022).
