"""Khmer Word Segmentation Studio: reusable, lossless Khmer NLP engine.

Features: UTF-8 dictionary, Unicode grapheme clusters, FMM/BMM/BiMM,
Viterbi-style SMART dictionary matching, optional pretrained CRF/neural models,
zero-width-space boundaries, stopwords, frequencies and unique vocabulary.

Educational design: important operations have inline comments. No sentence-specific
word preferences are secretly applied. Khmer segmentation is ambiguous; compare
predictions with independent annotated data rather than promising perfect results.
"""

from __future__ import annotations  # Allow forward references in type annotations.

import csv  # Read the original dictionary without pandas.
import io  # Handle uploaded dictionary CSV bytes.
import math  # Supply logarithmic-like model scores where needed.
import re  # Recognize URLs and email addresses before punctuation splitting.
import unicodedata  # Check Unicode categories and apply safe NFC normalization.
from collections import Counter  # Count word frequencies without deduplicating the sentence.
from dataclasses import dataclass  # Store structured token and result information.
from pathlib import Path  # Find the dictionary relative to this file.
from typing import Iterable  # Accept sets, lists or generators for stopwords.

import regex  # Support complete Unicode grapheme clusters through the \X pattern.

# Locate the default full dictionary, supplied with this project.
DEFAULT_DICTIONARY = Path(__file__).resolve().parent / "datasets" / "cleaned_khmer_dictionary.csv"

# Khmer users often put this invisible character at real word boundaries.
ZERO_WIDTH_SPACE = "\u200b"

# Normalize stopwords to a set: duplicates are automatically removed.
DEFAULT_STOPWORDS = frozenset({
    "ខ្ញុំ", "អ្នក", "គាត់", "យើង", "ពួកគេ", "ដែល", "និង", "ឬ", "ជា", "គឺ",
    "មាន", "បាន", "នឹង", "នៅ", "ទៅ", "មក", "ពី", "ក្នុង", "លើ", "ដោយ",
    "ដើម្បី", "សម្រាប់", "របស់", "កំពុង", "មិន", "ទេ", "ហើយ", "ក៏",
    "អាច", "ត្រូវ", "ថា", "ប៉ុន្តែ", "ព្រោះ", "តើ", "សូម", "ដល់",
    "មួយ", "នេះ", "នោះ", "ទាំង", "ជាមួយ", "ប្រសិនបើ", "ឲ្យ", "ឱ្យ",
})

# All supported algorithms; CRF/NEURAL require an optional installed package.
METHODS = ("fmm", "bmm", "bimm", "smart", "crf", "neural")

# These are word-like classes. Separators and punctuation are NOT word tokens.
WORD_KINDS = frozenset({"known", "unknown", "latin", "number", "other_word", "url", "email"})

# Keep an entire HTTP(S) URL together in mixed Khmer/English text.
URL_RE = re.compile(r"https?://[^\s\u200b]+", re.IGNORECASE)

# Preserve email addresses rather than splitting them at @ and dots.
EMAIL_RE = re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}")

# Compiled extended-grapheme regex prevents splitting a Khmer dependent mark off its base.
CLUSTER_RE = regex.compile(r"\X")

# U+17D7 is a Khmer repetition marker (category Lm), but not a normal word letter.
SPECIAL_PUNCTUATION = frozenset({"\u17d7"})

# Use a sentinel distinct from all grapheme strings to mark a complete word in a trie.
END = "\u0000"


def normalize(text: str) -> str:
    """Canonicalize Unicode without deleting any word boundaries or punctuation."""
    if not isinstance(text, str):  # Reject unexpected input types early.
        raise TypeError("Text must be a string.")
    return unicodedata.normalize("NFC", text)  # NFC preserves Khmer text meaning and ZWSP.


def clusters(text: str) -> list[str]:
    """Split a Khmer string into Unicode grapheme clusters, not individual code points."""
    return CLUSTER_RE.findall(normalize(text))  # \X retains consonants and attached marks.


def _punct(ch: str) -> bool:
    """Check punctuation, including the special Khmer repetition marker."""
    return ch in SPECIAL_PUNCTUATION or unicodedata.category(ch).startswith("P")


def _khmer(ch: str) -> bool:
    """Identify Khmer letters, combining marks and joiners in a Khmer word span."""
    category = unicodedata.category(ch)  # Unicode distinguishes letters from digits/signs.
    return (
        (0x1780 <= ord(ch) <= 0x17FF and category[:1] in {"L", "M"} and not _punct(ch))
        or ch in {"\u200c", "\u200d", "\u2060"}  # Joiners remain attached when present.
    )


def _clean_dictionary_word(value: str) -> str:
    """Accept only entries safe for ordinary word matching; keep source CSV unchanged."""
    word = normalize(value).strip()  # Trim exterior whitespace, but not marks.
    if not word or ZERO_WIDTH_SPACE in word:  # Avoid matching empty or already-separated text.
        return ""
    if any(ch.isspace() for ch in word):  # Spaced phrases are not ordinary word tokens.
        return ""
    if all(_punct(ch) for ch in word):  # Punctuation is handled separately by the scanner.
        return ""
    return word  # Preserve compounds, even when their POS is unavailable.


def load_dictionary(source: str | Path | bytes = DEFAULT_DICTIONARY) -> dict[str, str]:
    """Load Khmer word/POS CSV using UTF-8 or UTF-8 BOM, deduplicating by word."""
    if isinstance(source, bytes):  # Streamlit's uploaded files provide byte content.
        stream = io.StringIO(source.decode("utf-8-sig"))  # Strip UTF-8 BOM if present.
    else:  # Local files are accepted by absolute or relative path.
        stream = Path(source).open("r", encoding="utf-8-sig", newline="")
    vocabulary: dict[str, str] = {}  # Store a single POS value for each unique word.
    with stream:  # Close the file or memory stream safely.
        reader = csv.DictReader(stream)  # Use CSV headers rather than positional indices.
        fields = set(reader.fieldnames or [])  # Handle empty files gracefully.
        if "word" not in fields:  # POS column is optional for user dictionaries.
            raise ValueError("Dictionary CSV must contain a 'word' column; 'pos' is optional.")
        for row in reader:  # Read each source entry once.
            word = _clean_dictionary_word(row.get("word") or "")  # Validate lexicon key.
            pos = (row.get("pos") or "").strip()  # Keep a missing POS blank, never fabricate it.
            if not word:  # Skip punctuation-only and spaced phrases for dictionary matching.
                continue
            if word not in vocabulary or (not vocabulary[word] and pos):  # Preserve known POS.
                vocabulary[word] = pos  # Dictionary keys are unique by construction.
    if not vocabulary:  # Segmentation cannot work with an empty lexicon.
        raise ValueError("The dictionary contains no usable word entries.")
    return vocabulary  # The original CSV is never edited.


def dictionary_report(source: str | Path | bytes = DEFAULT_DICTIONARY) -> dict[str, int]:
    """Audit raw dictionary quality without changing the CSV's contents."""
    if isinstance(source, bytes):  # Handle a dictionary uploaded from Streamlit.
        stream = io.StringIO(source.decode("utf-8-sig"))
    else:  # Or use the project dictionary on disk.
        stream = Path(source).open("r", encoding="utf-8-sig", newline="")
    counts: Counter[str] = Counter()  # Collect transparent data-quality statistics.
    found: set[str] = set()  # Compare normalized keys to find duplicate occurrences.
    with stream:  # Always close the open stream.
        reader = csv.DictReader(stream)  # Parse quoted CSV fields correctly.
        for row in reader:  # Examine each source row, not only filtered entries.
            raw = row.get("word") or ""  # Treat missing word cells as empty strings.
            cleaned = normalize(raw.strip())  # Normalize only to compare identities.
            counts["rows"] += 1  # Count every CSV record, including blank entries.
            if not cleaned:  # Empty words never enter the matching trie.
                counts["empty_words"] += 1
                continue
            if cleaned in found:  # Track repeated normalized spellings.
                counts["duplicate_occurrences"] += 1
            found.add(cleaned)  # Retain the key for further duplicate detection.
            if not (row.get("pos") or "").strip():  # POS may be missing legitimately.
                counts["missing_pos"] += 1
            if raw != raw.strip():  # Flag surrounding whitespace separately.
                counts["surrounding_whitespace"] += 1
            if any(c.isspace() for c in cleaned):  # Includes non-breaking spaces U+00A0.
                counts["internal_whitespace"] += 1
            if ZERO_WIDTH_SPACE in cleaned:  # Flag phrases pre-segmented with ZWSP.
                counts["internal_zwsp"] += 1
            if all(_punct(c) for c in cleaned):  # Punctuation-only lexicon entries.
                counts["punctuation_only"] += 1
            if cleaned != raw.strip():  # Flag non-NFC raw forms.
                counts["non_nfc"] += 1
    counts["unique_words"] = len(found)  # Total distinct source spellings after NFC.
    counts["usable_words"] = len(load_dictionary(source))  # Count filtered matching entries.
    return dict(counts)  # Convert Counter to a simple report dictionary.


@dataclass(frozen=True)
class Token:
    """A token from the NFC-normalized text, with inclusive/exclusive offsets."""
    text: str  # Exact token spelling from the normalized text.
    kind: str  # Known, unknown, number, latin, punctuation, separator or symbol.
    start: int  # Starting character index of this token (inclusive).
    end: int  # Ending character index (exclusive).
    pos: str = ""  # POS label from supplied dictionary, if available.
    stopword: bool = False  # Stopword membership never removes text automatically.


@dataclass
class Result:
    """Reusable segmentation output for CLI, Streamlit, notebooks and CSV processing."""
    original: str  # Exact text before Unicode normalization.
    normalized: str  # NFC form used by the engine.
    method: str  # Selected algorithm.
    parts: list[Token]  # All spans, including punctuation, spaces and existing ZWSP.

    @property
    def tokens(self) -> list[Token]:
        """Return all word-like tokens, including repeated occurrences."""
        return [item for item in self.parts if item.kind in WORD_KINDS]

    @property
    def words(self) -> list[str]:
        """Return the full segmentation: duplicates are intentionally preserved."""
        return [item.text for item in self.tokens]

    @property
    def total_words(self) -> int:
        """Count every word occurrence (screenshot-style Total Words)."""
        return len(self.tokens)

    @property
    def unique_tokens(self) -> list[str]:
        """Return a separate unique vocabulary, preserving first occurrence order."""
        return list(dict.fromkeys(self.words))

    @property
    def frequencies(self) -> Counter[str]:
        """Count occurrences so duplicate words are visible in the analysis."""
        return Counter(self.words)

    @property
    def reconstructed(self) -> str:
        """Prove that segmentation has not deleted spaces, symbols or Khmer marks."""
        return "".join(part.text for part in self.parts)

    @property
    def zero_space(self) -> str:
        """Join only word tokens with U+200B, matching the notebook screenshot format."""
        return ZERO_WIDTH_SPACE.join(self.words)

    @property
    def zero_space_preserved(self) -> str:
        """Insert ZWSP between adjacent words, retaining original spaces/punctuation."""
        pieces: list[str] = []  # Build a lossless display sequence.
        prev: Token | None = None  # Remember whether the last part was a word.
        for item in self.parts:  # Traverse source order, including separators.
            if prev is not None and prev.kind in WORD_KINDS and item.kind in WORD_KINDS:
                pieces.append(ZERO_WIDTH_SPACE)  # Separate adjacent inferred words.
            pieces.append(item.text)  # Preserve the exact current part.
            prev = item  # Update the previous part.
        return "".join(pieces)  # Keep punctuation and source whitespace unchanged.

    def cleaned_tokens(
        self,
        remove_stopwords: bool = True,
        lowercase_latin: bool = True,
        include_numbers: bool = True,
    ) -> list[str]:
        """Return content tokens for vocabulary analysis, not sentence rewriting."""
        output: list[str] = []  # Cleaned tokens still include repeated occurrences.
        for token in self.tokens:  # Only process word-like kinds, never punctuation.
            if remove_stopwords and token.stopword:  # Optionally omit common function words.
                continue
            if not include_numbers and token.kind == "number":  # Optional number filtering.
                continue
            text = token.text.casefold() if lowercase_latin and token.kind == "latin" else token.text
            output.append(text)  # Preserve repeated occurrences for frequency counts.
        return output

    def vocabulary(
        self,
        remove_stopwords: bool = True,
        lowercase_latin: bool = True,
        include_numbers: bool = True,
    ) -> list[dict[str, object]]:
        """Return unique vocabulary rows with frequency counts and no duplicates."""
        clean = self.cleaned_tokens(remove_stopwords, lowercase_latin, include_numbers)
        counts = Counter(clean)  # Count each normalized content token.
        return [  # Preserve first-occurrence order using dict.fromkeys.
            {"token": word, "frequency": counts[word]}
            for word in dict.fromkeys(clean)
        ]

    def token_rows(self) -> list[dict[str, object]]:
        """Produce downloadable token records including POS and stopword flags."""
        counts = self.frequencies  # Compute the frequency map only once.
        return [  # Build one row per token occurrence, including duplicates.
            {"index": i, "token": tok.text, "kind": tok.kind, "pos": tok.pos,
             "stopword": tok.stopword, "start": tok.start, "end": tok.end,
             "frequency": counts[tok.text]}
            for i, tok in enumerate(self.tokens, 1)
        ]

    def show(self, detail: bool = False) -> None:
        """Print the four fields from the user's screenshot, with optional statistics."""
        print("Original Text:", self.original)  # Source sentence unchanged.
        print("Segment:", self.words)  # Word tokens including repeated words.
        print("ZeroSpace:", self.zero_space)  # Insert invisible U+200B boundaries.
        print("Total Words:", self.total_words)  # Count all token occurrences.
        if detail:  # Optional detailed analysis is separate from the four screenshot fields.
            print("Unique Tokens:", self.unique_tokens)
            print("Unique Count:", len(self.unique_tokens))
            print("Stopwords:", [x.text for x in self.tokens if x.stopword])
            print("Text Preserved:", self.reconstructed == self.normalized)


def _scan(text: str) -> list[tuple[str, str, int, int]]:
    """Losslessly pretokenize text into Khmer runs, script spans and separators."""
    output: list[tuple[str, str, int, int]] = []  # Each row stores kind, text and offsets.
    i = 0  # Current Unicode string index.
    n = len(text)  # Total character count after NFC normalization.
    while i < n:  # Read until no characters remain.
        start = i  # Remember this chunk's beginning.
        ch = text[i]  # Examine the current character.
        url = URL_RE.match(text, i)  # Recognize URLs before punctuation.
        email = EMAIL_RE.match(text, i)  # Recognize email addresses before punctuation.
        if url:  # Keep URLs intact as one token.
            i = url.end()
            kind = "url"
        elif email:  # Keep emails intact as one token.
            i = email.end()
            kind = "email"
        elif ch == ZERO_WIDTH_SPACE:  # Explicit zero-width boundaries cannot be crossed.
            while i < n and text[i] == ZERO_WIDTH_SPACE:
                i += 1
            kind = "separator"
        elif ch.isspace():  # Preserve whitespace, including newlines and NBSP.
            while i < n and text[i].isspace():
                i += 1
            kind = "separator"
        elif unicodedata.category(ch) == "Nd":  # Group Khmer and ASCII digits.
            i += 1
            while i < n:  # Accept decimal points only between two digits.
                nxt = text[i]
                if unicodedata.category(nxt) == "Nd":
                    i += 1
                elif nxt in {".", ","} and i + 1 < n and unicodedata.category(text[i + 1]) == "Nd":
                    i += 1
                else:
                    break
            kind = "number"
        elif _punct(ch):  # Separate symbols and punctuation from word matching.
            i += 1
            while i < n and _punct(text[i]):
                i += 1
            kind = "punctuation"
        elif _khmer(ch):  # Collect a contiguous Khmer script span for segmentation.
            i += 1
            while i < n and _khmer(text[i]):
                i += 1
            kind = "khmer"
        elif unicodedata.category(ch).startswith("L"):  # Handle Latin and other scripts.
            i += 1
            while i < n and (unicodedata.category(text[i])[:1] in {"L", "M"} or text[i] == "_"):
                if _khmer(text[i]):  # Do not merge Latin and Khmer spans.
                    break
                i += 1
            kind = "latin" if ch.isascii() else "other_word"
        else:  # Emoji and other symbols remain visible but are not word tokens.
            match = CLUSTER_RE.match(text, i)  # Consume an entire Unicode grapheme.
            i = match.end() if match else i + 1
            kind = "symbol"
        output.append((kind, text[start:i], start, i))  # Preserve exact text and offsets.
    return output  # The concatenation of all chunks equals the NFC-normalized input.


def _trie_insert(root: dict, word_clusters: list[str], pos: str) -> None:
    """Insert a dictionary word into the grapheme-cluster trie."""
    node = root  # Start with the root dictionary.
    for part in word_clusters:  # Traverse one grapheme at a time.
        node = node.setdefault(part, {})  # Create a child node if absent.
    node[END] = pos  # Mark the complete word and store its POS value.


class KhmerSegmenter:
    """Dictionary-based Maximum Matching and optional learned Khmer segmenters."""

    def __init__(
        self,
        vocabulary: dict[str, str],
        stopwords: Iterable[str] = DEFAULT_STOPWORDS,
        max_word_clusters: int | None = None,
    ) -> None:
        # Use None to allow the full maximum dictionary length: no arbitrary truncation.
        if max_word_clusters is not None and max_word_clusters < 1:
            raise ValueError("max_word_clusters must be positive or None.")
        self.vocabulary = {word: pos for word, pos in vocabulary.items() if _clean_dictionary_word(word) == word}
        if not self.vocabulary:  # Reject empty vocabularies.
            raise ValueError("Segmentation requires a nonempty dictionary.")
        self.stopwords = frozenset(normalize(w).strip() for w in stopwords if str(w).strip())
        self.forward: dict = {}  # Trie for left-to-right maximum matching.
        self.backward: dict = {}  # Trie for right-to-left maximum matching.
        longest = 0  # Track the true maximum word length in the dictionary.
        for word, pos in self.vocabulary.items():  # Build each trie once, not per sentence.
            pieces = clusters(word)  # Match over graphemes, not individual vowel marks.
            _trie_insert(self.forward, pieces, pos)  # Forward prefix trie.
            _trie_insert(self.backward, list(reversed(pieces)), pos)  # Reversed suffix trie.
            longest = max(longest, len(pieces))  # Update the observed dictionary maximum.
        self.dictionary_max_clusters = longest  # Expose the actual maximum matching length.
        self.max_word_clusters = min(longest, max_word_clusters or longest)  # Optional user cap.

    def _matches(self, chars: list[str], start: int, backward: bool = False) -> list[tuple[int, str, str]]:
        """Return all dictionary candidates from a position, shortest to longest."""
        hits: list[tuple[int, str, str]] = []  # (end/start, word, pos) for each hit.
        node = self.backward if backward else self.forward  # Select direction-specific trie.
        limit = self.max_word_clusters  # Bound traversal by the actual dictionary length.
        for step in range(limit):  # A missing trie edge ends the search early.
            index = start - 1 - step if backward else start + step
            if index < 0 or index >= len(chars):
                break
            piece = chars[index]  # Read one orthographic cluster.
            if piece not in node:
                break
            node = node[piece]  # Advance to the next trie node.
            if END in node:  # A full word ends at this position.
                end = start - step - 1 if backward else start + step + 1
                word = "".join(chars[end:start]) if backward else "".join(chars[start:end])
                hits.append((end, word, node[END]))  # Keep all candidates for DP.
        return hits

    @staticmethod
    def _merge_unknown(tokens: list[tuple[str, str, str]]) -> list[tuple[str, str, str]]:
        """Merge adjacent unmatched graphemes into readable unknown spans."""
        merged: list[tuple[str, str, str]] = []  # Avoid one-token-per-diacritic errors.
        for word, kind, pos in tokens:  # Examine the segmentation path in order.
            if merged and kind == "unknown" and merged[-1][1] == "unknown":
                merged[-1] = (merged[-1][0] + word, "unknown", "")
            else:
                merged.append((word, kind, pos))  # Preserve known words unchanged.
        return merged

    def _fmm(self, chars: list[str]) -> list[tuple[str, str, str]]:
        """Forward Maximum Matching: always choose the longest lexicon prefix."""
        i = 0  # Start at the left edge.
        result: list[tuple[str, str, str]] = []  # Store token, kind and POS.
        while i < len(chars):
            options = self._matches(chars, i)  # Find all valid prefixes here.
            if options:  # Longest match is the final option.
                end, word, pos = options[-1]
                result.append((word, "known", pos))
                i = end  # Continue after the recognized word.
            else:
                result.append((chars[i], "unknown", ""))  # Preserve unmatched cluster.
                i += 1
        return self._merge_unknown(result)  # Group unknown spans for readability.

    def _bmm(self, chars: list[str]) -> list[tuple[str, str, str]]:
        """Backward Maximum Matching: longest lexicon suffix from right to left."""
        i = len(chars)  # Start at the right edge.
        reversed_result: list[tuple[str, str, str]] = []
        while i > 0:
            options = self._matches(chars, i, backward=True)  # Find matching suffixes.
            if options:
                begin, word, pos = options[-1]  # Longest suffix wins.
                reversed_result.append((word, "known", pos))
                i = begin  # Move left beyond the matched suffix.
            else:
                reversed_result.append((chars[i - 1], "unknown", ""))
                i -= 1
        return self._merge_unknown(list(reversed(reversed_result)))  # Restore reading order.

    @staticmethod
    def _rank(tokens: list[tuple[str, str, str]]) -> tuple[int, int, int, int]:
        """Rank BiMM paths: fewer unknown clusters/tokens/singletons, more POS tags."""
        unknown = sum(len(clusters(w)) for w, kind, _ in tokens if kind == "unknown")
        single = sum(len(clusters(w)) == 1 for w, _, _ in tokens)
        tagged = sum(bool(pos) for _, kind, pos in tokens if kind == "known")
        return (unknown, len(tokens), single, -tagged)

    def _bimm(self, chars: list[str]) -> list[tuple[str, str, str]]:
        """Bidirectional MM: compare FMM and BMM by consistent tie-breaking rules."""
        forward = self._fmm(chars)  # Candidate from the left.
        backward = self._bmm(chars)  # Candidate from the right.
        return min((forward, backward), key=self._rank)  # Prefer FMM on equal rank.

    def _smart(self, chars: list[str]) -> list[tuple[str, str, str]]:
        """Global dynamic programming over dictionary paths; no sentence exceptions."""
        n = len(chars)  # Count available grapheme clusters.
        scores = [-math.inf] * (n + 1)  # Best path score for each suffix position.
        best: list[tuple[int, str, str, str] | None] = [None] * (n + 1)
        scores[n] = 0.0  # The empty suffix contributes no cost.
        for i in range(n - 1, -1, -1):  # Viterbi-style backward search.
            scores[i] = -3.5 + scores[i + 1]  # Fallback unknown cluster cost.
            best[i] = (i + 1, chars[i], "unknown", "")  # Preserve unmatched text.
            for end, word, pos in self._matches(chars, i):  # Consider every dictionary edge.
                width = end - i  # Word length in grapheme clusters.
                # Score candidates using generic length and POS evidence, not example-specific rules.
                word_score = 2.0 * width - 2.0 + (1.7 if pos else -0.7)
                if width == 1:
                    word_score -= 1.2  # Discourage one-cluster fragments where better choices exist.
                if width >= 5 and not pos:
                    word_score -= 0.30 * (width - 4)  # Avoid trusting very long untagged phrases.
                if word in self.stopwords:
                    word_score += 0.3  # Give a small grammatical function-word prior.
                score = word_score + scores[end]  # Combine this edge with the best suffix.
                if score > scores[i] + 1e-9:  # Keep only a better global path.
                    scores[i] = score
                    best[i] = (end, word, "known", pos)
        output: list[tuple[str, str, str]] = []  # Reconstruct the best complete path.
        i = 0
        while i < n:
            choice = best[i]  # Read the chosen edge for this position.
            assert choice is not None  # Every position has the unknown fallback.
            end, word, kind, pos = choice
            output.append((word, kind, pos))
            i = end
        return self._merge_unknown(output)  # Return stable word spans.

    def _learned(self, text: str, method: str) -> list[tuple[str, str, str]]:
        """Use optional khmercut pretrained models; do not add manual word preferences."""
        try:  # Import lazily; dictionary matching works with no pretrained dependencies.
            if method == "crf":
                from khmercut import tokenize as tokenizer  # CRF model supplied by package.
            else:
                from khmercut.nn import tokenize as tokenizer  # Optional ONNX neural model.
        except ImportError as exc:  # Give users an actionable installation message.
            raise RuntimeError("Install the optional models with: python -m pip install 'khmercut[nn]'") from exc
        try:  # Run the learned model on the Khmer run, not punctuation or source whitespace.
            words = list(tokenizer(text))
        except UnicodeDecodeError as exc:  # Windows default cp1252 breaks the neural vocabulary.
            raise RuntimeError("On Windows, start Python in UTF-8 mode: python -X utf8 -m streamlit run app.py") from exc
        if not all(isinstance(x, str) and x for x in words) or "".join(words) != text:
            raise ValueError("The pretrained model did not preserve the original Khmer characters.")
        return [(word, "known" if word in self.vocabulary else "unknown", self.vocabulary.get(word, ""))
                for word in words]  # Attach POS where available, without inventing labels.

    def _run(self, text: str, method: str) -> list[tuple[str, str, str]]:
        """Select the requested algorithm for a contiguous Khmer text run."""
        if method in {"crf", "neural"}:  # Learned models operate directly on Unicode strings.
            return self._learned(text, method)
        chars = clusters(text)  # Maximum Matching and SMART operate on grapheme clusters.
        return {"fmm": self._fmm, "bmm": self._bmm, "bimm": self._bimm, "smart": self._smart}[method](chars)

    def segment(self, text: str, method: str = "smart") -> Result:
        """Segment arbitrary Khmer/Latin/digit text without destroying source characters."""
        if method not in METHODS:  # Reject misspelled algorithm names clearly.
            raise ValueError(f"method must be one of {METHODS}")
        original = text  # Keep the original string for display and auditing.
        normalized = normalize(text)  # Work with NFC Unicode for consistent dictionary lookup.
        parts: list[Token] = []  # Build exact spans across the complete normalized string.
        for kind, chunk, start, end in _scan(normalized):  # Preserve punctuation and separators.
            if kind != "khmer":  # Non-Khmer spans are not passed to Khmer word segmentation.
                parts.append(Token(chunk, kind, start, end, stopword=(chunk in self.stopwords)))
                continue
            cursor = start  # Track the current offset within this Khmer run.
            for word, word_kind, pos in self._run(chunk, method):  # Predict its word boundaries.
                finish = cursor + len(word)  # Offsets use Unicode code points, not bytes.
                parts.append(Token(word, word_kind, cursor, finish, pos, word in self.stopwords))
                cursor = finish  # Advance to the next predicted token.
            if cursor != end:  # Models must account for every original Unicode character.
                raise AssertionError("Segmentation failed to cover a Khmer text span.")
        result = Result(original, normalized, method, parts)  # Create reusable output.
        if result.reconstructed != normalized:  # Enforce lossless segmentation invariant.
            raise AssertionError("A segmentation step changed or discarded input text.")
        for item in result.parts:  # Validate all token offsets against the source.
            if normalized[item.start:item.end] != item.text:
                raise AssertionError("Token position does not match its source text.")
        return result  # The application may then clean tokens without altering segmentation.


# Public convenience API for notebooks and one-line scripts.
def segment_text(
    text: str,
    method: str = "smart",
    dictionary_path: str | Path | bytes = DEFAULT_DICTIONARY,
    stopwords: Iterable[str] = DEFAULT_STOPWORDS,
) -> Result:
    """Load a dictionary, construct the engine and segment a user-provided string."""
    vocabulary = load_dictionary(dictionary_path)  # Read source dictionary once for this call.
    engine = KhmerSegmenter(vocabulary, stopwords=stopwords)  # Build the trie and stopword set.
    return engine.segment(text, method=method)  # Return a reusable result with .show().


if __name__ == "__main__":  # Allow easy terminal testing without Streamlit.
    import argparse  # Parse command-line flags only when run directly.
    parser = argparse.ArgumentParser(description="Khmer Word Segmentation Studio")
    parser.add_argument("--text", default="ទឹកត្រីដែលឆ្ងាញ់គឺធ្វើមកពីត្រីដែលរស់នៅក្នុងទឹកសាប")
    parser.add_argument("--method", choices=METHODS, default="smart")
    parser.add_argument("--interactive", action="store_true")
    parser.add_argument("--compare", action="store_true")
    parser.add_argument("--dictionary", default=str(DEFAULT_DICTIONARY))
    args = parser.parse_args()  # Read the user's CLI choices.
    text = input("Enter Khmer text / បញ្ចូលអត្ថបទខ្មែរ: ") if args.interactive else args.text
    engine = KhmerSegmenter(load_dictionary(args.dictionary))  # Reuse for comparison.
    for choice in METHODS if args.compare else (args.method,):
        print("\n---", choice.upper(), "---")
        try:  # Optional learned models may not be installed.
            engine.segment(text, choice).show(detail=True)
        except (RuntimeError, ValueError) as exc:
            print("Unavailable:", exc)
