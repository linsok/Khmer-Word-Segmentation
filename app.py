
"""Professional Streamlit application for Khmer Word Segmentation Studio.

Run: python -X utf8 -m streamlit run app.py
Uses the same engine as the notebook, CLI and tests.
User-provided text is escaped before being inserted into HTML.
"""

from __future__ import annotations

import csv
import html
import hashlib
import importlib.util
import json
from pathlib import Path

import pandas as pd
import streamlit as st

from csv_tools import batch_segment, load_csv, output_csv
from segmenter import (
    DEFAULT_DICTIONARY,
    DEFAULT_STOPWORDS,
    METHODS,
    ZERO_WIDTH_SPACE,
    KhmerSegmenter,
    Result,
    dictionary_report,
    load_dictionary,
)


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Khmer Word Segmentation Studio",
    layout="wide",
)


# ============================================================
# APPLICATION STYLING
# ============================================================

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&family=Noto+Sans+Khmer:wght@400;500;600;700&display=swap');

:root { color-scheme: light; }

html, body, [class*="css"] {
    font-family: 'DM Sans', 'Noto Sans Khmer', 'Khmer OS', sans-serif;
}

.stApp, [data-testid="stAppViewContainer"] {
    background: #f5f8fc;
    color: #193049;
}

.block-container {
    max-width: 1350px;
    padding-top: 1.8rem;
    padding-bottom: 3rem;
}

header[data-testid="stHeader"] {
    background: rgba(245, 248, 252, .96);
}

/* SIDEBAR */

section[data-testid="stSidebar"] {
    background: #f8fafd;
    border-right: 1px solid #dae4ef;
}

section[data-testid="stSidebar"] [data-testid="stSidebarContent"] {
    padding: 1.3rem .85rem 1.5rem;
}

section[data-testid="stSidebar"] [data-testid="stVerticalBlockBorderWrapper"] {
    border-radius: 16px;
}

section[data-testid="stSidebar"] [data-testid="stVerticalBlockBorderWrapper"] > div {
    background: #fff;
}

section[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p,
section[data-testid="stSidebar"] .stWidgetLabel p,
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] label p {
    color: #304761 !important;
    font-weight: 650 !important;
}

section[data-testid="stSidebar"] [data-testid="stCaptionContainer"] p,
section[data-testid="stSidebar"] [data-testid="stCaptionContainer"] {
    color: #64748b !important;
}

section[data-testid="stSidebar"] [data-baseweb="select"] > div,
section[data-testid="stSidebar"] [data-baseweb="input"] > div,
section[data-testid="stSidebar"] [data-baseweb="textarea"] > div,
section[data-testid="stSidebar"] input,
section[data-testid="stSidebar"] textarea {
    background: #fff !important;
    color: #1e3857 !important;
    border-color: #cbd8e6 !important;
}

section[data-testid="stSidebar"] [data-baseweb="select"] span,
section[data-testid="stSidebar"] [data-baseweb="select"] div {
    color: #1e3857 !important;
}

section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] {
    background: #f8fbff;
    border: 1px dashed #b8cadf;
    border-radius: 12px;
}

section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] * {
    color: #38526e !important;
}

section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] button {
    background: #fff !important;
    border: 1px solid #bbccdf !important;
    color: #234367 !important;
}

section[data-testid="stSidebar"] [data-testid="stExpander"] {
    border: 1px solid #e0e9f2;
    border-radius: 12px;
    background: #fff;
}

section[data-testid="stSidebar"] [data-testid="stExpander"] summary,
section[data-testid="stSidebar"] [data-testid="stExpander"] summary p {
    color: #274462 !important;
    font-weight: 650 !important;
}

.sidebar-brand {
    padding: 9px 8px 15px;
}

.sidebar-kicker {
    color: #307789;
    font-size: 10px;
    font-weight: 800;
    text-transform: uppercase;
    letter-spacing: .16em;
}

.sidebar-brand-title {
    font-size: 21px;
    font-weight: 800;
    letter-spacing: -.035em;
    color: #172f4d;
    padding: 4px 0;
}

.sidebar-brand-detail {
    color: #667c93;
    font-size: 12px;
    line-height: 1.6;
}

.sidebar-section-title {
    color: #1b3755;
    font-size: 15px;
    font-weight: 800;
    margin: 0 0 3px;
}

.sidebar-section-description {
    font-size: 12px;
    line-height: 1.65;
    color: #657a92;
    margin-bottom: 8px;
}

.sidebar-status {
    border-top: 1px solid #e3ebf3;
    margin-top: 10px;
    padding-top: 14px;
}

.sidebar-status-dot {
    display: inline-block;
    width: 8px;
    height: 8px;
    border-radius: 100%;
    background: #228b6e;
    margin-right: 8px;
}

.sidebar-status-title {
    color: #1a3d52;
    font-size: 13px;
    font-weight: 750;
}

.sidebar-status-subtitle {
    color: #687d93;
    margin: 4px 0 0 16px;
    font-size: 12px;
}

.sidebar-footer {
    font-size: 11px;
    color: #70839a;
    padding: 4px 8px 8px;
    line-height: 1.65;
}

/* MAIN CONTENT */

.hero {
    padding: 32px 36px 34px;
    background: linear-gradient(120deg, #132844 0%, #1b456b 70%, #166979 100%);
    color: #fff;
    border-radius: 20px;
    box-shadow: 0 16px 40px rgba(25, 55, 91, .12);
    position: relative;
    overflow: hidden;
    margin-bottom: 23px;
}

.hero:after {
    content: '';
    position: absolute;
    top: -150px;
    right: -130px;
    width: 320px;
    height: 320px;
    border: 45px solid rgba(255, 255, 255, .045);
    border-radius: 50%;
    pointer-events: none;
}

.eyebrow {
    font-size: 10px;
    font-weight: 800;
    letter-spacing: .18em;
    text-transform: uppercase;
    color: #92d9d1;
    margin-bottom: 14px;
}

.hero h1 {
    color: #fff !important;
    font-size: clamp(27px, 3vw, 42px);
    font-weight: 800;
    letter-spacing: -.035em;
    margin: 0 0 14px;
}

.hero p {
    font-size: 14px;
    color: #d7eaf4;
    line-height: 1.85;
    margin: 0;
    max-width: 750px;
}

.section {
    color: #1d3858;
    font-weight: 800;
    font-size: 19px;
    margin: 22px 0 6px;
}

.muted {
    color: #61758e;
    font-size: 13px;
    line-height: 1.75;
}

.flow-card {
    background: #fff;
    border: 1px solid #dce6f0;
    border-radius: 15px;
    padding: 21px 22px;
    line-height: 2.7;
    min-height: 86px;
    box-shadow: 0 4px 18px rgba(32, 55, 86, .035);
    word-break: break-word;
    font-family: 'Noto Sans Khmer', 'Khmer OS', sans-serif;
}

.token {
    display: inline-block;
    margin: 3px 5px 3px 0;
    padding: 3px 10px 4px;
    line-height: 1.9;
    border-radius: 9px;
    font-size: 17px;
    border: 1px solid transparent;
}

.token.known {
    background: #e5f4eb;
    border-color: #b0d7c1;
    color: #185d45;
}

.token.unknown {
    background: #fff3e5;
    border-color: #efc899;
    color: #875017;
}

.token.latin, .token.other_word, .token.url, .token.email {
    background: #eaf1ff;
    border-color: #c5d8f0;
    color: #204d85;
}

.token.number {
    background: #f1edfc;
    border-color: #d4c7ef;
    color: #5c4690;
}

.token.punctuation, .token.symbol {
    background: #f0f3f7;
    border-color: #dce4eb;
    color: #60758a;
}

.token.stopword {
    box-shadow: inset 0 -3px 0 #8977cb;
}

.space {
    display: inline-block;
    min-width: 8px;
}

.zwsp {
    color: #675294;
    background: #eee9f8;
    font-size: 10px;
    padding: 2px 5px;
    border-radius: 4px;
}

.card-note {
    border: 1px solid #dbe5f1;
    padding: 14px 17px;
    background: #fff;
    border-radius: 12px;
    color: #536a82;
    font-size: 13px;
    line-height: 1.8;
}

[data-testid="stMetric"] {
    background: #fff;
    border: 1px solid #dde7f2;
    border-radius: 14px;
    padding: 14px 18px;
}

[data-testid="stMetricLabel"],
[data-testid="stMetricLabel"] p {
    color: #526c86 !important;
    font-weight: 650;
}

[data-testid="stMetricValue"] {
    color: #1a385a;
    font-weight: 800;
}

[data-testid="stAppViewContainer"] [data-testid="stWidgetLabel"] p {
    color: #314b68 !important;
    font-weight: 650;
}

[data-testid="stAppViewContainer"] [data-testid="stCaptionContainer"] p {
    color: #647891 !important;
}

textarea {
    font-family: 'Noto Sans Khmer', 'Khmer OS', sans-serif !important;
    font-size: 17px !important;
    line-height: 2 !important;
    background: #fff !important;
    color: #1c344f !important;
    border-color: #cbd8e6 !important;
}

.stTabs [data-baseweb="tab-list"] {
    gap: 15px;
    border-bottom: 1px solid #dce6ef;
}

.stTabs [data-baseweb="tab"] {
    font-weight: 700;
    color: #4f647b !important;
}

.stTabs [aria-selected="true"] {
    color: #135b88 !important;
}

div.stButton > button[kind="primary"],
div.stDownloadButton > button[kind="primary"] {
    background: #185a95 !important;
    color: #fff !important;
    font-weight: 750;
    border: 1px solid #185a95 !important;
    border-radius: 10px;
    min-height: 43px;
}

div.stButton > button[kind="primary"] p,
div.stDownloadButton > button[kind="primary"] p {
    color: #fff !important;
}

div.stButton > button[kind="secondary"],
div.stDownloadButton > button[kind="secondary"] {
    background: #fff !important;
    color: #214562 !important;
    font-weight: 650;
    border: 1px solid #c4d3e2 !important;
    border-radius: 10px;
    min-height: 41px;
}

div.stButton > button[kind="secondary"] p,
div.stDownloadButton > button[kind="secondary"] p {
    color: #214562 !important;
}

div.stButton > button:hover,
div.stDownloadButton > button:hover {
    border-color: #286aa1 !important;
}

.footer {
    border-top: 1px solid #dce5ef;
    margin-top: 30px;
    padding-top: 17px;
    font-size: 12px;
    color: #778ba0;
    text-align: center;
}

@media (max-width: 760px) {
    .block-container { padding: 1rem .8rem; }
    .hero { padding: 24px 20px; border-radius: 14px; }
    .flow-card { padding: 14px 12px; }
    .token { font-size: 15px; }
}
</style>
"""

st.markdown(CSS, unsafe_allow_html=True)


# ============================================================
# EXAMPLE SENTENCES AND ALGORITHMS
# ============================================================

EXAMPLES = {
    "Fish sauce sentence": "ទឹកត្រីដែលឆ្ងាញ់គឺធ្វើមកពីត្រីដែលរស់នៅក្នុងទឹកសាប",
    "Learning Khmer": "ខ្ញុំស្រឡាញ់ភាសាខ្មែរ",
    "Everyday sentence": "កុមារកំពុងលេងបាល់។ ខ្ញុំទៅសាលារៀន!",
    "Zero-width spaces": "ខ្ញុំ\u200bស្រឡាញ់\u200bភាសាខ្មែរ",
    "Numbers and mixed scripts": "ថ្ងៃនេះខ្ញុំរៀន Python 3 និងមាន១២,៥០០រៀល។",
    "Multiple sentences": "សិស្សអានសៀវភៅ។ គាត់ទៅសាលារៀន។ សួស្តីបងប្អូន!",
}

METHOD_LABELS = {
    "fmm": "Forward Maximum Matching (FMM)",
    "bmm": "Backward Maximum Matching (BMM)",
    "bimm": "Bidirectional Maximum Matching (BiMM)",
    "smart": "Smart Dictionary Scoring",
    "crf": "CRF - Pretrained Khmer Model",
    "neural": "Neural - Pretrained ONNX Model",
}


# ============================================================
# CACHED NLP RESOURCES
# ============================================================

@st.cache_resource(
    show_spinner="Loading Khmer dictionary…",
    max_entries=3,
)
def cached_engine(
    dictionary_bytes: bytes,
    stopword_values: tuple[str, ...],
    max_clusters: int | None,
) -> KhmerSegmenter:
    """Load the trie only when the dictionary or options change."""
    lexicon = load_dictionary(dictionary_bytes)

    return KhmerSegmenter(
        lexicon,
        stopwords=stopword_values,
        max_word_clusters=max_clusters,
    )


@st.cache_data(show_spinner=False)
def cached_dictionary_report(dictionary_bytes: bytes) -> dict[str, int]:
    """Cache the quality report for uploaded dictionary data."""
    return dictionary_report(dictionary_bytes)


# ============================================================
# TOKEN PRESENTATION AND EXPORT
# ============================================================

def to_token_html(result: Result, show_boundaries: bool) -> str:
    """Create safe, highlighted HTML for segmented words."""
    chunks: list[str] = []

    for item in result.parts:
        if item.kind == "separator":
            if ZERO_WIDTH_SPACE in item.text and show_boundaries:
                chunks.append('<span class="zwsp">ZWSP</span>')

            if any(ch.isspace() for ch in item.text):
                chunks.append('<span class="space">&ensp;</span>')

            continue

        role = html.escape(item.kind, quote=True)
        suffix = " stopword" if item.stopword else ""
        label = html.escape(item.text)

        chunks.append(
            f'<span class="token {role}{suffix}">{label}</span>'
        )

    return '<div class="flow-card">' + "".join(chunks) + "</div>"


def result_exports(
    result: Result,
    remove_stops: bool,
    latin_lower: bool,
    numbers: bool,
) -> None:
    """Provide CSV and JSON download controls."""
    token_table = result.token_rows()
    vocab_table = result.vocabulary(
        remove_stops,
        latin_lower,
        numbers,
    )

    token_fields = [
        "index", "token", "kind", "pos",
        "stopword", "start", "end", "frequency",
    ]

    columns = st.columns(3)

    with columns[0]:
        st.download_button(
            "Download tokens CSV",
            output_csv(token_fields, token_table),
            file_name="segmented_tokens.csv",
            mime="text/csv",
            use_container_width=True,
        )

    with columns[1]:
        st.download_button(
            "Download vocabulary CSV",
            output_csv(["token", "frequency"], vocab_table),
            file_name="unique_vocabulary.csv",
            mime="text/csv",
            use_container_width=True,
        )

    with columns[2]:
        payload = {
            "original": result.original,
            "method": result.method,
            "words": result.words,
            "zero_space": result.zero_space,
            "total_words": result.total_words,
            "unique_tokens": result.unique_tokens,
            "vocabulary": vocab_table,
            "all_parts": [
                vars(part) for part in result.parts
            ],
        }

        st.download_button(
            "Download JSON",
            json.dumps(
                payload,
                ensure_ascii=False,
                indent=2,
            ).encode("utf-8"),
            file_name="khmer_segmentation.json",
            mime="application/json",
            use_container_width=True,
        )


def show_result(
    result: Result,
    remove_stops: bool,
    latin_lower: bool,
    numbers: bool,
    boundaries: bool,
) -> None:
    """Display segmentation, statistics and detailed analysis."""
    cleaned = result.cleaned_tokens(
        remove_stops,
        latin_lower,
        numbers,
    )

    unique_cleaned = list(dict.fromkeys(cleaned))

    columns = st.columns(4)
    columns[0].metric("Total word tokens", result.total_words)
    columns[1].metric("Unique tokens", len(result.unique_tokens))
    columns[2].metric("Clean vocabulary", len(unique_cleaned))
    columns[3].metric(
        "Stopwords in text",
        sum(t.stopword for t in result.tokens),
    )

    st.markdown(
        '<div class="section">Segmentation results</div>',
        unsafe_allow_html=True,
    )

    st.code(
        f"Original Text: {result.original}\n"
        f"Segment: {result.words}\n"
        f"ZeroSpace: {result.zero_space}\n"
        f"Total Words: {result.total_words}",
        language="text",
    )

    st.markdown(
        '<div class="section">Segmented sentence</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        to_token_html(result, boundaries),
        unsafe_allow_html=True,
    )

    st.caption(
        "Green = dictionary word · Orange = unknown word · "
        "Purple underline = stopword. "
        "Repeated words remain in the original segmentation."
    )

    tab_words, tab_vocab, tab_text, tab_meta = st.tabs(
        [
            "Word tokens",
            "Unique vocabulary",
            "ZeroSpace output",
            "Details & POS",
        ]
    )

    with tab_words:
        st.write(result.words)
        st.caption(
            "Total Words counts all occurrences, including repetitions."
        )
        st.dataframe(
            pd.DataFrame(result.token_rows()),
            use_container_width=True,
            hide_index=True,
        )

    with tab_vocab:
        table = result.vocabulary(
            remove_stops,
            latin_lower,
            numbers,
        )

        st.dataframe(
            pd.DataFrame(
                table,
                columns=["token", "frequency"],
            ),
            use_container_width=True,
            hide_index=True,
        )

        st.caption(
            "Stopwords are excluded only when enabled. "
            "Each vocabulary token appears once."
        )

    with tab_text:
        st.text_area(
            "Segmented with U+200B separators",
            value=result.zero_space,
            height=110,
            disabled=True,
        )

        st.text_area(
            "Original separators and punctuation preserved",
            value=result.zero_space_preserved,
            height=110,
            disabled=True,
        )

        st.caption(
            "U+200B is an invisible zero-width-space separator."
        )

    with tab_meta:
        st.json({
            "method": result.method,
            "total_words": result.total_words,
            "unique_words": len(result.unique_tokens),
            "content_tokens": len(cleaned),
            "clean_vocabulary": len(unique_cleaned),
            "unknown": sum(
                t.kind == "unknown" for t in result.tokens
            ),
            "unicode_preserved": (
                result.reconstructed == result.normalized
            ),
        })

        st.caption(
            "POS labels come from the dictionary. "
            "Missing POS does not make a word invalid."
        )

    st.markdown(
        '<div class="section">Export your analysis</div>',
        unsafe_allow_html=True,
    )

    result_exports(
        result,
        remove_stops,
        latin_lower,
        numbers,
    )


# ============================================================
# APPLICATION HEADER
# ============================================================

st.markdown(
    """
    <div class="hero">
        <div class="eyebrow">
            NLP LAB / KHMER LANGUAGE PROCESSING
        </div>
        <h1>Khmer Word Segmentation Studio</h1>
        <p>
            បំបែកពាក្យខ្មែរ · Maximum Matching ·
            CRF &amp; Neural Models · Zero-Width Spaces ·
            Vocabulary Analysis
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        '<div class="sidebar-brand">'
        '<div class="sidebar-kicker">KHMER NLP WORKSPACE</div>'
        '<div class="sidebar-brand-title">Segmentation Studio</div>'
        '<div class="sidebar-brand-detail">'
        'Configure the model, analyze vocabulary and manage your lexicon.'
        '</div>'
        '</div>',
        unsafe_allow_html=True,
    )

    pretrained_available = (
        importlib.util.find_spec("khmercut") is not None
    )

    default_method = (
        "crf" if pretrained_available else "smart"
    )

    # Segmentation configuration.
    with st.container(border=True):
        st.markdown(
            '<div class="sidebar-section-title">'
            'Segmentation settings'
            '</div>',
            unsafe_allow_html=True,
        )

        st.markdown(
            '<div class="sidebar-section-description">'
            'Choose how Khmer word boundaries are detected.'
            '</div>',
            unsafe_allow_html=True,
        )

        method = st.selectbox(
            "Segmentation method",
            METHODS,
            index=METHODS.index(default_method),
            format_func=lambda code: METHOD_LABELS[code],
            help=(
                "FMM, BMM, BiMM and SMART use the dictionary. "
                "CRF and Neural require khmercut."
            ),
        )

        if (
            method in {"crf", "neural"}
            and not pretrained_available
        ):
            st.warning(
                "Optional package missing. Install: "
                "python -m pip install 'khmercut[nn]'"
            )

        st.caption(
            "Model selection applies to text input and CSV batches."
        )

    # Vocabulary processing options.
    with st.container(border=True):
        st.markdown(
            '<div class="sidebar-section-title">'
            'Vocabulary cleanup'
            '</div>',
            unsafe_allow_html=True,
        )

        st.markdown(
            '<div class="sidebar-section-description">'
            'Choose what appears in the unique vocabulary, '
            'not what remains in the sentence.'
            '</div>',
            unsafe_allow_html=True,
        )

        remove_stops = st.checkbox(
            "Remove stopwords from vocabulary",
            value=True,
            help=(
                "Stopwords remain in the original segmented sentence; "
                "they are filtered only from vocabulary analysis."
            ),
        )

        latin_lower = st.checkbox(
            "Lowercase Latin vocabulary",
            value=True,
        )

        include_numbers = st.checkbox(
            "Include numbers in vocabulary",
            value=True,
        )

        show_boundaries = st.checkbox(
            "Show original zero-width boundaries",
            value=True,
        )

        with st.expander("Edit stopword list", expanded=False):
            st.caption(
                "One word per line. "
                "Duplicate entries are automatically removed."
            )

            stopword_text = st.text_area(
                "Stopwords",
                value="\n".join(sorted(DEFAULT_STOPWORDS)),
                height=175,
                key="stopwords_editor",
            )

        stopwords = tuple(sorted({
            word.strip()
            for word in stopword_text.splitlines()
            if word.strip()
        }))

    # Dictionary management.
    with st.container(border=True):
        st.markdown(
            '<div class="sidebar-section-title">'
            'Dictionary management'
            '</div>',
            unsafe_allow_html=True,
        )

        st.markdown(
            '<div class="sidebar-section-description">'
            'Use the built-in Khmer dictionary or upload your own lexicon.'
            '</div>',
            unsafe_allow_html=True,
        )

        dictionary_upload = st.file_uploader(
            "Dictionary CSV (optional)",
            type="csv",
            key="lexicon_csv",
            help=(
                "CSV must contain a 'word' column; "
                "'pos' is optional."
            ),
        )

        dictionary_bytes = (
            dictionary_upload.getvalue()
            if dictionary_upload is not None
            else DEFAULT_DICTIONARY.read_bytes()
        )

        auto_max = st.checkbox(
            "Use full dictionary maximum length",
            value=True,
            help=(
                "Maximum Matching searches the full word length "
                "found in your lexicon."
            ),
        )

        max_clusters = (
            None
            if auto_max
            else st.number_input(
                "Max grapheme clusters per word",
                min_value=1,
                max_value=200,
                value=25,
            )
        )

        try:
            engine = cached_engine(
                dictionary_bytes,
                stopwords,
                max_clusters,
            )

            st.markdown(
                '<div class="sidebar-status">'
                '<span class="sidebar-status-dot"></span>'
                f'<span class="sidebar-status-title">'
                f'{len(engine.vocabulary):,} matching words ready'
                '</span>'
                f'<div class="sidebar-status-subtitle">'
                f'Longest word: {engine.dictionary_max_clusters} '
                'grapheme clusters'
                '</div>'
                '</div>',
                unsafe_allow_html=True,
            )

        except (ValueError, UnicodeDecodeError) as exc:
            st.error(f"Dictionary error: {exc}")
            st.stop()

        settings_signature = (
            hashlib.sha256(dictionary_bytes).hexdigest(),
            stopwords,
            max_clusters,
            method,
        )

        with st.expander(
            "Dictionary quality report",
            expanded=False,
        ):
            st.json(
                cached_dictionary_report(dictionary_bytes)
            )

    st.markdown(
        '<div class="sidebar-footer">'
        'Local word segmentation. Input text is not sent '
        'to an external text-processing API.'
        '</div>',
        unsafe_allow_html=True,
    )


# ============================================================
# MAIN NAVIGATION
# ============================================================

text_tab, csv_tab, eval_tab, about_tab = st.tabs(
    [
        "Text Studio",
        "CSV Batch",
        "Evaluation",
        "How it works",
    ]
)


# ============================================================
# TEXT STUDIO
# ============================================================

with text_tab:

    st.markdown(
        '<div class="section">Enter Khmer text</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="muted">'
        'Paste Khmer words, sentences, paragraphs, punctuation, '
        'digits or mixed English. '
        'The segmenter accepts input without visible spaces.'
        '</div>',
        unsafe_allow_html=True,
    )

    if "text_input" not in st.session_state:
        st.session_state.text_input = (
            EXAMPLES["Fish sauce sentence"]
        )

    sample = st.selectbox(
        "Try an example",
        list(EXAMPLES),
        label_visibility="collapsed",
    )

    if st.button(
        "Load selected example",
        key="load_example",
        use_container_width=False,
    ):
        st.session_state.text_input = EXAMPLES[sample]
        st.rerun()

    user_text = st.text_area(
        "Khmer text",
        key="text_input",
        height=150,
        placeholder="សូមបញ្ចូលអត្ថបទខ្មែរ...",
    )

    if st.button(
        "Segment text",
        type="primary",
        key="segment_main",
        use_container_width=True,
    ):
        if not user_text.strip():
            st.warning(
                "Please enter a Khmer sentence before segmenting."
            )

        else:
            try:
                with st.spinner(
                    "Finding Khmer word boundaries…"
                ):
                    st.session_state.single_result = (
                        engine.segment(
                            user_text,
                            method=method,
                        )
                    )

                    st.session_state.single_settings = (
                        settings_signature
                    )

            except (
                RuntimeError,
                ValueError,
                AssertionError,
            ) as exc:
                st.error(str(exc))
                st.session_state.single_result = None

    current = st.session_state.get("single_result")

    if current is not None and (
        current.original != user_text
        or st.session_state.get("single_settings")
        != settings_signature
    ):
        st.info(
            "The sentence or segmentation settings changed. "
            "Select 'Segment text' to refresh."
        )
        current = None

    if current is not None:
        show_result(
            current,
            remove_stops,
            latin_lower,
            include_numbers,
            show_boundaries,
        )

        with st.expander(
            "Compare the segmentation algorithms",
            expanded=False,
        ):
            st.caption(
                "Algorithms can disagree on compound boundaries. "
                "Use annotated data to evaluate correctness."
            )

            if st.button(
                "Run comparison",
                key="run_comparison",
            ):
                comparison: list[
                    dict[str, object]
                ] = []

                for algorithm in METHODS:
                    if (
                        algorithm in {"crf", "neural"}
                        and not pretrained_available
                    ):
                        continue

                    try:
                        prediction = engine.segment(
                            user_text,
                            algorithm,
                        )

                        comparison.append({
                            "algorithm": algorithm.upper(),
                            "words": prediction.total_words,
                            "unique": len(
                                prediction.unique_tokens
                            ),
                            "tokens": " | ".join(
                                prediction.words
                            ),
                        })

                    except (
                        RuntimeError,
                        ValueError,
                    ) as exc:
                        comparison.append({
                            "algorithm": algorithm.upper(),
                            "error": str(exc),
                        })

                st.dataframe(
                    pd.DataFrame(comparison),
                    use_container_width=True,
                    hide_index=True,
                )


# ============================================================
# CSV BATCH PROCESSING
# ============================================================

with csv_tab:

    st.markdown(
        '<div class="section">'
        'Import Khmer text from a CSV'
        '</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="muted">'
        'Upload a UTF-8 or UTF-16 CSV, choose which column '
        'contains Khmer text, then export processed rows '
        'and unique vocabulary. Original columns and row '
        'order remain intact.'
        '</div>',
        unsafe_allow_html=True,
    )

    input_csv = st.file_uploader(
        "Upload sentence CSV",
        type=["csv"],
        key="sentences_csv",
    )

    if input_csv is not None:
        try:
            delimiter_choice = st.selectbox(
                "CSV delimiter",
                ["auto", ",", ";", "tab", "|"],
                key="csv_delimiter",
            )

            table = load_csv(
                input_csv.getvalue(),
                delimiter_choice,
            )

            st.caption(
                f"Detected {len(table.rows):,} rows · "
                f"{len(table.fields)} columns · "
                f"delimiter {table.delimiter!r}"
            )

            text_column = st.selectbox(
                "Which column contains Khmer text?",
                table.fields,
                key="text_column",
            )

            st.dataframe(
                pd.DataFrame(
                    table.rows[:12],
                    columns=table.fields,
                ),
                use_container_width=True,
                hide_index=True,
            )

            if len(table.rows) > 25000:
                st.warning(
                    "More than 25,000 rows. "
                    "Use the included cli.py for large batches."
                )

            if st.button(
                "Process CSV rows",
                type="primary",
                key="process_csv",
                disabled=(len(table.rows) > 25000),
            ):
                try:
                    with st.spinner(
                        "Segmenting imported Khmer sentences…"
                    ):
                        fields, enriched, vocab = batch_segment(
                            table,
                            text_column,
                            engine,
                            method=method,
                            remove_stopwords=remove_stops,
                            lowercase_latin=latin_lower,
                            include_numbers=include_numbers,
                        )

                    st.session_state.batch_result = (
                        fields,
                        enriched,
                        vocab,
                    )

                    st.session_state.batch_source = (
                        hashlib.sha256(
                            input_csv.getvalue()
                        ).hexdigest(),
                        text_column,
                        settings_signature,
                        remove_stops,
                        latin_lower,
                        include_numbers,
                    )

                except (
                    RuntimeError,
                    ValueError,
                    AssertionError,
                ) as exc:
                    st.error(
                        f"CSV processing failed: {exc}"
                    )

            batch = st.session_state.get(
                "batch_result"
            )

            batch_key = (
                hashlib.sha256(
                    input_csv.getvalue()
                ).hexdigest(),
                text_column,
                settings_signature,
                remove_stops,
                latin_lower,
                include_numbers,
            )

            if (
                batch is not None
                and st.session_state.get("batch_source")
                != batch_key
            ):
                st.info(
                    "Your CSV or settings changed. "
                    "Select 'Process CSV rows' to refresh exports."
                )
                batch = None

            if batch is not None:
                fields, enriched, vocab = batch

                st.markdown(
                    '<div class="section">'
                    'Processed CSV output'
                    '</div>',
                    unsafe_allow_html=True,
                )

                x, y, z = st.columns(3)

                x.metric(
                    "Processed rows",
                    len(enriched),
                )

                y.metric(
                    "Total word tokens",
                    sum(
                        int(row["seg_total_words"])
                        for row in enriched
                    ),
                )

                z.metric(
                    "Corpus vocabulary",
                    len(vocab),
                )

                st.dataframe(
                    pd.DataFrame(
                        enriched[:250],
                        columns=fields,
                    ),
                    use_container_width=True,
                    hide_index=True,
                )

                st.caption(
                    "Showing the first 250 rows. "
                    "The CSV download contains all processed rows."
                )

                col_a, col_b = st.columns(2)

                with col_a:
                    st.download_button(
                        "Download processed CSV",
                        output_csv(
                            fields,
                            enriched,
                        ),
                        file_name="khmer_segmented_rows.csv",
                        mime="text/csv",
                        use_container_width=True,
                    )

                with col_b:
                    st.download_button(
                        "Download corpus vocabulary",
                        output_csv(
                            ["token", "frequency"],
                            vocab,
                        ),
                        file_name="khmer_corpus_vocabulary.csv",
                        mime="text/csv",
                        use_container_width=True,
                    )

                with st.expander(
                    "Corpus vocabulary preview",
                    expanded=False,
                ):
                    st.dataframe(
                        pd.DataFrame(
                            vocab[:250]
                        ),
                        use_container_width=True,
                        hide_index=True,
                    )

        except (
            ValueError,
            UnicodeError,
            csv.Error,
        ) as exc:
            st.error(
                f"Could not read the CSV: {exc}"
            )


# ============================================================
# SEGMENTATION EVALUATION
# ============================================================

with eval_tab:

    st.markdown(
        '<div class="section">'
        'Evaluate segmentation accuracy'
        '</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="card-note">'
        'Accuracy needs a human-annotated reference. '
        'A matching word count alone is not sufficient. '
        'The five supplied examples are demonstration data, '
        'not an independent estimate of real-world performance.'
        '</div>',
        unsafe_allow_html=True,
    )

    st.write("")

    st.code(
        "python -X utf8 tests/evaluate.py",
        language="powershell",
    )

    st.caption(
        "The evaluator reports boundary precision, recall, "
        "F1 and exact sentence matches. "
        "You can add rows to tests/gold_sentences.csv."
    )

    if st.button(
        "Evaluate included gold sentences",
        key="evaluate_button",
    ):
        from evaluation import evaluate

        try:
            gold = (
                Path(__file__).resolve().parent
                / "tests"
                / "gold_sentences.csv"
            )

            rows, summary = evaluate(
                engine,
                gold,
                methods=(
                    "fmm",
                    "bmm",
                    "bimm",
                    "smart",
                ),
            )

            st.dataframe(
                pd.DataFrame(summary),
                use_container_width=True,
                hide_index=True,
            )

            with st.expander(
                "Sentence-level predictions",
                expanded=False,
            ):
                st.dataframe(
                    pd.DataFrame(rows),
                    use_container_width=True,
                    hide_index=True,
                )

        except (
            ValueError,
            OSError,
        ) as exc:
            st.error(str(exc))


# ============================================================
# DOCUMENTATION TAB
# ============================================================

with about_tab:

    st.markdown(
        '<div class="section">'
        'How the pipeline works'
        '</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
**Input → Normalize (NFC) → Detect Khmer/Latin/numbers/ZWSP → Segment Khmer words → Label stopwords → Clean for vocabulary → Count unique tokens.**

**Maximum Matching (FMM/BMM/BiMM)** chooses the longest dictionary match from one or both directions.

**SMART** compares multiple dictionary paths with dynamic programming.

**CRF and Neural** use optional pretrained Khmer sequence models from `khmercut`.

- **Total Words:** Includes repeated words because they occur in the sentence.
- **Unique Tokens:** Includes each segmented spelling once.
- **Vocabulary:** The unique set after optional stopword removal and other cleanup.
- **ZeroSpace:** Joins detected words with Unicode `U+200B`.
- **Punctuation and whitespace:** Preserved in the original and lossless output but excluded from the word vocabulary.

Khmer word boundaries can be ambiguous, especially for compounds. Different methods may produce different segmentations. Compare predictions with human-annotated reference sentences rather than forcing a particular word count.

**Research inspiration:** [Mao Makara's KhmerWordSegmentation](https://github.com/MaoMakara/KhmerWordSegmentation). This application is an independent implementation.
"""
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    '<div class="footer">'
    'Khmer Word Segmentation Studio · '
    'Dictionary-based and pretrained methods · '
    'Local processing'
    '</div>',
    unsafe_allow_html=True,
)
