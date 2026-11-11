"""Build the fully annotated Jupyter notebook from version-controlled text cells."""
from pathlib import Path
import nbformat as nbf

cells=[]

def md(s):
    cells.append(nbf.v4.new_markdown_cell(s.strip()))

def code(s):
    cells.append(nbf.v4.new_code_cell(s.strip()))

md("""
# 🇰🇭 Khmer Word Segmentation — Complete NLP Notebook

This notebook uses the **same `segmenter.py` engine** as the Streamlit app, with the 101,134-entry original dictionary.

**Methods:** Forward Maximum Matching (FMM), Backward Maximum Matching (BMM), BiMM, SMART dynamic programming, plus optional pretrained CRF/Neural segmentation.

**Outputs:** Original Text · Segment · ZeroSpace (`U+200B`) · Total Words · unique tokens · deduplicated stopwords · frequencies · POS · CSV batch processing · boundary F1 evaluation.

> **Accuracy:** Different methods may split compounds differently. We do not force an example to return 11 words. Gold annotations are for evaluation only.
""")

md("""
## 1. Import the libraries and find project folders

The notebook works whether Jupyter starts inside the project root or `notebooks/`. Comments describe every meaningful Python statement.
""")
code(r'''
# Import sys so Python can find segmenter.py in the project root.
import sys
# Import Path so Windows and Linux paths work the same way.
from pathlib import Path
# Import Counter for token frequencies.
from collections import Counter
# Import pandas to display word tables.
import pandas as pd
# Import the standard CSV writer for optional exports.
import csv
# Detect the current Jupyter working directory.
PROJECT_ROOT = Path.cwd()
# Move to the parent if Jupyter started inside notebooks/.
if not (PROJECT_ROOT / "segmenter.py").exists():
    PROJECT_ROOT = PROJECT_ROOT.parent
# Add the project folder to Python's import path.
sys.path.insert(0, str(PROJECT_ROOT))
# Import our shared segmentation code and constants.
from segmenter import KhmerSegmenter, DEFAULT_STOPWORDS, DEFAULT_DICTIONARY, ZERO_WIDTH_SPACE, load_dictionary, dictionary_report, clusters
# Import CSV functions that are also used by the web application.
from csv_tools import load_csv, batch_segment, output_csv
# Import evaluation functions that never train on the gold examples.
from evaluation import evaluate
# Print the detected project folder to help with troubleshooting.
print("Project root:", PROJECT_ROOT)
''')

md("""
## 2. Load and inspect the Khmer dictionary

Dictionary CSV headers: `word` and `pos`. The original data is not modified. Blank POS values are allowed. Spaced phrases and punctuation-only entries are excluded from the **matching trie**, not erased from the source CSV.
""")
code('''
# Point to the original dictionary supplied with the project.
DICTIONARY_PATH = PROJECT_ROOT / "datasets" / "cleaned_khmer_dictionary.csv"
# Read, normalize, and deduplicate usable lexicon entries.
word_dictionary = load_dictionary(DICTIONARY_PATH)
# Build the segmentation trie once for repeated sentence processing.
segmenter = KhmerSegmenter(word_dictionary)
# Print the number of valid unique entries in the matching trie.
print("Matching dictionary entries:", len(word_dictionary))
# Print the maximum observed dictionary word length in grapheme clusters.
print("Longest word (clusters):", segmenter.dictionary_max_clusters)
# Audit the raw original dictionary without altering it.
print("Source quality report:", dictionary_report(DICTIONARY_PATH))
''')

md("""
## 3. Unicode, Khmer clusters and zero-width spaces

Khmer grapheme clusters combine consonants with dependent vowel marks and subscript consonants. `U+200B` is an invisible but explicit boundary.
""")
code('''
# Create a sample with two existing zero-width separators.
zwsp_sentence = "ខ្ញុំ\\u200bស្រឡាញ់\\u200bភាសាខ្មែរ"
# Explain the character clusters that form the Khmer word ខ្ញុំ.
print("Khmer clusters:", clusters("ខ្ញុំ"))
# Count the invisible boundaries in the input.
print("Original zero-width spaces:", zwsp_sentence.count(ZERO_WIDTH_SPACE))
# Segment while preserving the explicit boundaries.
zwsp_result = segmenter.segment(zwsp_sentence, method="smart")
# Display the screenshot-style output fields.
zwsp_result.show()
# Prove that no Unicode characters were lost while segmenting.
print("Text preserved:", zwsp_result.reconstructed == zwsp_result.normalized)
''')

md("""
## 4. First full Khmer segmentation — `sentence.show()`

Use **SMART** as the offline scored-dictionary baseline. The notebook's `.show()` output follows your screenshot's four headings. Note that dictionary methods can disagree about compound words.
""")
code('''
# Define the example sentence supplied in your homework assignment.
text = "ទឹកត្រីដែលឆ្ងាញ់គឺធ្វើមកពីត្រីដែលរស់នៅក្នុងទឹកសាប"
# Run the automatic SMART dictionary search (no preferred-word flags).
sentence = segmenter.segment(text, method="smart")
# Display the exact four fields in the reference screenshot.
sentence.show()
# Show a full unique-token list separately from the sentence.
print("Unique Tokens:", sentence.unique_tokens)
# Show a frequency map that reveals repeated words.
print("Frequencies:", dict(sentence.frequencies))
''')

md("""
## 5. Forward, backward and bidirectional Maximum Matching

**FMM** chooses the longest dictionary word starting at the left edge. **BMM** starts from the right. **BiMM** compares them using unknown-token and token-count heuristics. These are baseline NLP algorithms, not perfect linguistic oracles.
""")
code('''
# Define the dictionary-based methods we want to compare.
methods = ["fmm", "bmm", "bimm", "smart"]
# Process the same input with each algorithm.
for method in methods:
    # Predict word boundaries using the selected algorithm.
    result = segmenter.segment(text, method=method)
    # Print the method name and its predicted segmentation.
    print(f"{method.upper():5s} | {result.total_words:2d} words | {result.words}")
''')

md("""
## 6. Vocabulary pipeline — tokenize → normalize Latin case → remove stopwords → unique tokens

**Critical distinction:** Repeated words must remain in the segmented sentence. Deduplication applies only to the *vocabulary*. We do **not** delete stopwords before finding word boundaries. Punctuation and whitespace never enter the word-token vocabulary.
""")
code('''
# Collect word occurrences from the segmented sentence, preserving duplicates.
all_tokens = sentence.words
# Build a set of stopwords; a set cannot contain duplicate entries.
stopwords = set(DEFAULT_STOPWORDS)
# Obtain cleaned word occurrences after removing stopwords.
content_tokens = sentence.cleaned_tokens(remove_stopwords=True, lowercase_latin=True)
# Create a unique vocabulary while preserving first occurrence order.
unique_vocabulary = list(dict.fromkeys(content_tokens))
# Count the frequency of cleaned word occurrences.
word_frequencies = Counter(content_tokens)
# Display the three distinct outputs and counts.
print("All word tokens:", all_tokens)
print("Cleaned content tokens:", content_tokens)
print("Unique vocabulary:", unique_vocabulary)
print("Number of unique vocabulary tokens:", len(unique_vocabulary))
print("No duplicate stopwords in set:", len(stopwords) == len(set(stopwords)))
''')

md("""
## 7. Vocabulary frequency and POS tables

The imported `pos` column may be blank. We display those blanks rather than guessing a grammatical label.
""")
code('''
# Convert token occurrences and their POS values into a display table.
token_table = pd.DataFrame(sentence.token_rows())
# Convert unique vocabulary words into a frequency table.
vocab_table = pd.DataFrame(sentence.vocabulary(remove_stopwords=True))
# Display token-level annotations.
display(token_table)
# Display unique vocabulary entries and frequencies.
display(vocab_table)
''')

md("""
## 8. User-defined sentence input

To try *any* new Khmer sentence, change the string below and rerun this cell. An optional `input()` line is included but commented, so automated notebook execution never blocks.
""")
code('''
# Replace this string with any new Khmer sentence you want to analyze.
user_sentence = "ខ្ញុំស្រឡាញ់ភាសាខ្មែរ"
# For interactive terminal-style input, uncomment the following line.
# user_sentence = input("Enter Khmer text: ")
# Segment the new input with the same engine.
user_result = segmenter.segment(user_sentence, method="smart")
# Print the original sentence, tokens, U+200B output and word count.
user_result.show(detail=True)
''')

md("""
## 9. Punctuation, whitespace, Khmer digits and mixed English

The engine keeps the original text intact. Cleaning removes punctuation *from vocabulary analysis* only, without deleting it from the source or pretending punctuation is a word.
""")
code('''
# Define a mixed-script Khmer example with punctuation and Khmer digits.
mixed_text = "ថ្ងៃនេះខ្ញុំរៀន Python 3 និងមាន១២,៥០០រៀល។"
# Run dictionary-based segmentation.
mixed_result = segmenter.segment(mixed_text, method="smart")
# Show the word-like tokens without punctuation or spaces.
print("Word tokens:", mixed_result.words)
# Show the original-preserving normalized text.
print("Original text preserved:", mixed_result.reconstructed == mixed_result.normalized)
# Show zero-width spaces inserted between adjacent words while keeping punctuation.
print("ZeroSpace preserving punctuation:", mixed_result.zero_space_preserved)
# Show a clean unique vocabulary, including numbers.
print("Vocabulary:", mixed_result.vocabulary(remove_stopwords=True))
''')

md("""
## 10. Batch import from any CSV column

The CSV does not have to use the header `text`. Select the actual column, such as `khmer_sentence`. All original columns, empty cells, row order and record counts are retained in the processed output.
""")
code('''
# Find the demonstration CSV file in the examples folder.
INPUT_CSV = PROJECT_ROOT / "examples" / "input_sentences.csv"
# Read the file as bytes to demonstrate the same upload path as Streamlit.
table = load_csv(INPUT_CSV.read_bytes())
# Display the available columns so a user can choose the right text field.
print("CSV columns:", table.fields)
# Select the column containing Khmer sentences (not a hardcoded requirement in the engine).
text_column = "khmer_sentence"
# Process every row and build a corpus-wide unique vocabulary.
fields, processed_rows, corpus_vocabulary = batch_segment(
    table, text_column, segmenter, method="smart", remove_stopwords=True
)
# Display the processed row count and the unique vocabulary count.
print("Processed rows:", len(processed_rows))
print("Corpus unique tokens:", len(corpus_vocabulary))
# Show a compact output preview without deleting the original source columns.
display(pd.DataFrame(processed_rows).head())
# Show the corpus vocabulary with token frequencies.
display(pd.DataFrame(corpus_vocabulary).head(15))
''')

md("""
## 11. Export the processed CSV and vocabulary CSV

UTF-8 BOM encoding makes Khmer easier to open correctly in Microsoft Excel. This cell writes only to the project `examples/` folder and never overwrites the original input.
""")
code('''
# Define a new destination for the processed rows.
output_rows = PROJECT_ROOT / "examples" / "notebook_segmented_rows.csv"
# Define a separate destination for the unique corpus vocabulary.
output_vocab = PROJECT_ROOT / "examples" / "notebook_unique_vocabulary.csv"
# Export all original columns plus segmentation columns.
output_rows.write_bytes(output_csv(fields, processed_rows))
# Export unique tokens exactly once, with their frequencies.
output_vocab.write_bytes(output_csv(["token", "frequency"], corpus_vocabulary))
# Show where to find the generated result files.
print("Saved rows:", output_rows)
print("Saved vocabulary:", output_vocab)
''')

md("""
## 12. Optional pretrained Khmer models (CRF and Neural)

These require `khmercut[nn]`. They are optional in this notebook; the four Maximum Matching/dictionary methods run entirely offline using your original dictionary. On Windows, launch Jupyter with `PYTHONUTF8=1` to prevent the neural model's vocabulary from being decoded with cp1252.
""")
code('''
# Define the pretrained algorithms to demonstrate, if installed.
pretrained = ["crf", "neural"]
# Test each model without any sentence-specific word preferences.
for method in pretrained:
    try:
        # Ask the pretrained model to segment the same example sentence.
        learned_result = segmenter.segment(text, method=method)
        # Display predicted words and the automatically counted total.
        print(method.upper(), learned_result.words, learned_result.total_words)
    except (RuntimeError, ValueError, ImportError) as error:
        # Give an installation or UTF-8 note rather than crashing the notebook.
        print(f"{method.upper()} unavailable:", error)
''')

md("""
## 13. Accuracy evaluation: gold boundaries, precision, recall and F1

Reference annotations should be independently labeled by people. The included five examples are educational and are **not used to train** these methods. An exact word count does not guarantee exact word boundaries.
""")
code('''
# Locate the annotated reference sentences.
gold_csv = PROJECT_ROOT / "tests" / "gold_sentences.csv"
# Evaluate all four dictionary algorithms on the same five sentences.
per_sentence, overall = evaluate(segmenter, gold_csv, methods=("fmm", "bmm", "bimm", "smart"))
# Display micro-averaged precision, recall, F1 and exact sentence matches.
display(pd.DataFrame(overall))
# Display detailed sentence-level errors for review.
display(pd.DataFrame(per_sentence))
''')

md("""
## 14. Interpreting the 11-word screenshot

The user's human annotation of `ទឹកត្រីដែលឆ្ងាញ់គឺធ្វើមកពីត្រីដែលរស់នៅក្នុងទឹកសាប` contains **11 word occurrences** and **10 unique words** (because `ដែល` repeats). Longest Matching and pretrained models may differ at compound boundaries. We evaluate predictions against the reference instead of creating hidden hardcoded `--prefer-word` rules.
""")
code('''
# Record the reference segmentation only for display/evaluation.
gold_words = ["ទឹកត្រី", "ដែល", "ឆ្ងាញ់", "គឺ", "ធ្វើ", "មកពី", "ត្រី", "ដែល", "រស់នៅ", "ក្នុង", "ទឹកសាប"]
# Confirm this reference matches the raw text character-for-character.
assert "".join(gold_words) == text
# Count all word occurrences in the human annotation.
print("Reference total words:", len(gold_words))
# Count distinct words without changing the human annotation.
print("Reference unique words:", len(set(gold_words)))
# Print the reference only as a gold standard, not as a model prediction.
print("Reference annotation:", gold_words)
''')

md("""
## 15. Conclusions and next experiments

- **FMM/BMM/BiMM** provide a clear Maximum Matching baseline and support complete dictionary word lengths.
- **SMART** uses a global dictionary path score without special sentence-level preferences.
- **CRF/Neural** can use contextual patterns learned from external Khmer corpora, but still make mistakes on compounds.
- **Stopwords, punctuation and whitespace** should be removed for vocabulary analysis *after* tokenization, not erased before finding word boundaries.
- **Unique vocabulary** is a set-like result with one entry per word; the original sentence retains repeats.
- To measure general accuracy, expand the annotated corpus with unseen sentences, check tokenization conventions, and keep training data separate from evaluation data.

**Reference:** [MaoMakara/KhmerWordSegmentation](https://github.com/MaoMakara/KhmerWordSegmentation) — research inspiration; this implementation is independently written and not copied from the repository.
""")

nb=nbf.v4.new_notebook(cells=cells,metadata={"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},
"language_info":{"name":"python","version":"3.10"}})
output=Path(__file__).resolve().parent / "notebooks" / "khmer_segmentation.ipynb"
nbf.validate(nb)
nbf.write(nb,output)
print("Created",output,"with",len(cells),"cells")
