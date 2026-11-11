"""Safe CSV import, batch Khmer segmentation and UTF-8-BOM output.

Educational comments explain each major operation. The importer does not assume
that the input text is stored in a column named 'text': users can select any header.
"""

from __future__ import annotations  # Keep modern type hints compatible with Python 3.10+.

import csv  # Parse CSV and write properly quoted outputs.
import io  # Read uploaded bytes without creating temporary files.
from collections import Counter  # Calculate corpus-wide token frequencies.
from dataclasses import dataclass  # Package parsed CSV data cleanly.

from segmenter import KhmerSegmenter  # Share the exact same segmentation engine.

# Add output columns with a prefix to distinguish them from original data.
NEW_COLUMNS = (
    "seg_algorithm", "seg_tokens", "seg_zero_space", "seg_total_words",
    "seg_unique_words", "seg_content_tokens", "seg_vocabulary",
    "seg_vocabulary_size", "seg_stopword_occurrences", "seg_unknown_words",
)


@dataclass
class CSVTable:
    """A table with original column order and row contents preserved."""
    fields: list[str]  # Original headers as they appeared in the uploaded CSV.
    rows: list[dict[str, str]]  # Original values, including empty cells.
    delimiter: str = ","  # Actual or user-selected CSV delimiter.


def _decode(data: bytes) -> str:
    """Decode UTF-8/UTF-8-SIG and UTF-16 CSV without silently corrupting Khmer."""
    if data.startswith((b"\xff\xfe", b"\xfe\xff")):  # Detect a UTF-16 BOM first.
        return data.decode("utf-16")  # Python honors the BOM's endianness.
    try:  # UTF-8-SIG also accepts normal UTF-8 without a BOM.
        return data.decode("utf-8-sig")
    except UnicodeDecodeError as exc:  # Never silently replace invalid Khmer bytes.
        raise ValueError("CSV must be UTF-8 or UTF-16. Re-save it using UTF-8 encoding.") from exc


def load_csv(data: bytes, delimiter: str = "auto") -> CSVTable:
    """Read a user CSV with flexible headers, delimiter autodetection and blank lines."""
    text = _decode(data)  # Decode source bytes before CSV parsing.
    text = text.lstrip("\ufeff\r\n")  # Skip a possible BOM and blank lines before headers.
    if not text.strip():  # Reject empty CSV files clearly.
        raise ValueError("The uploaded CSV is empty.")
    if delimiter == "auto":  # Detect comma/semicolon/tab/pipe from the first rows.
        try:
            dialect = csv.Sniffer().sniff(text[:8192], delimiters=",;\t|")
            selected = dialect.delimiter
        except csv.Error:  # Fall back to comma for short or irregular files.
            selected = ","
    else:  # Allow users to override detection, including tab-separated files.
        selected = "\t" if delimiter == "tab" else delimiter
        if selected not in {",", ";", "|", "\t"}:
            raise ValueError("Unsupported CSV delimiter.")
    reader = csv.DictReader(io.StringIO(text, newline=""), delimiter=selected)
    fields = reader.fieldnames or []  # Read actual headers, not guessed labels.
    if not fields or any(not field.strip() for field in fields):
        raise ValueError("CSV requires a nonempty header for every column.")
    if len(fields) != len(set(fields)):  # Prevent accidental overwrites from duplicate headers.
        raise ValueError("CSV column headers must be unique.")
    rows: list[dict[str, str]] = []  # Preserve row order and original values.
    for row_index, row in enumerate(reader, start=2):
        if None in row:  # This occurs when a CSV row contains extra unquoted delimiters.
            raise ValueError(f"CSV row {row_index} contains more values than headers. Check quoting/delimiter.")
        rows.append({field: row.get(field) or "" for field in fields})
    return CSVTable(fields=fields, rows=rows, delimiter=selected)


def output_csv(fields: list[str], rows: list[dict[str, object]]) -> bytes:
    """Encode a table for downloading; UTF-8 BOM helps Excel display Khmer correctly."""
    buffer = io.StringIO(newline="")  # Use an in-memory CSV writer.
    writer = csv.DictWriter(buffer, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()  # Include a clear header row.
    writer.writerows(rows)  # Quote commas and Khmer text when required.
    return buffer.getvalue().encode("utf-8-sig")  # Preserve Khmer when opened in Excel.


def batch_segment(
    table: CSVTable,
    text_column: str,
    engine: KhmerSegmenter,
    method: str = "smart",
    remove_stopwords: bool = True,
    lowercase_latin: bool = True,
    include_numbers: bool = True,
) -> tuple[list[str], list[dict[str, object]], list[dict[str, object]]]:
    """Append segmentation output to EVERY original row and build a corpus vocabulary."""
    if text_column not in table.fields:  # Require an existing user-selected text column.
        raise ValueError(f"Column not found: {text_column!r}")
    fields = list(table.fields)  # Preserve original headers first.
    for name in NEW_COLUMNS:  # Prevent collisions with user-provided field names.
        if name in fields:
            raise ValueError(f"Input CSV already has a reserved output field: {name}")
        fields.append(name)  # Append derived fields without renaming originals.
    enriched: list[dict[str, object]] = []  # Store all processed CSV rows.
    corpus_counts: Counter[str] = Counter()  # Count the entire input corpus vocabulary.
    for row in table.rows:  # Process blank cells too, preserving row count.
        text = row.get(text_column, "")  # Read the original text cell exactly.
        result = engine.segment(text, method=method)  # Reuse the shared Khmer engine.
        clean = result.cleaned_tokens(remove_stopwords, lowercase_latin, include_numbers)
        unique_content = list(dict.fromkeys(clean))  # Deduplicate after optional filtering.
        corpus_counts.update(clean)  # Include repetitions in the corpus frequency.
        enriched_row: dict[str, object] = dict(row)  # Copy all original columns unchanged.
        enriched_row.update({  # Add transparent, spreadsheet-friendly outputs.
            "seg_algorithm": method,
            "seg_tokens": " | ".join(result.words),  # Readable full segmentation with duplicates.
            "seg_zero_space": result.zero_space,  # Invisible U+200B between tokens.
            "seg_total_words": result.total_words,  # Full word count, not unique count.
            "seg_unique_words": len(result.unique_tokens),  # Unique before stopword filtering.
            "seg_content_tokens": " | ".join(clean),  # Tokens after requested cleanup.
            "seg_vocabulary": " | ".join(unique_content),  # No duplicate tokens.
            "seg_vocabulary_size": len(unique_content),  # Distinct cleaned tokens.
            "seg_stopword_occurrences": sum(t.stopword for t in result.tokens),
            "seg_unknown_words": sum(t.kind == "unknown" for t in result.tokens),
        })
        enriched.append(enriched_row)  # Keep the source row order identical.
    corpus_vocab = [  # Produce corpus-wide unique tokens and their frequencies.
        {"token": token, "frequency": frequency}
        for token, frequency in corpus_counts.most_common()
    ]
    return fields, enriched, corpus_vocab  # Both tables can be downloaded as CSV.
