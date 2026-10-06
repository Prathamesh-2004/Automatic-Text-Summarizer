"""
app.py
------
MODULE 7 – USER INTERFACE (Streamlit)

Run from the project root:
    streamlit run app.py

The browser opens at http://localhost:8501
"""

import os

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

import html
import time

import pandas as pd
import streamlit as st

import config
from evaluation.metrics import compute_rouge, compute_stats
from preprocessing.text_cleaner import split_title, word_count
from summarization.abstractive_summarizer import abstractive_summary, load_model
from summarization.base import ArticleTooShortError
from summarization.frequency_summarizer import frequency_summary
from summarization.lead3_baseline import lead_summary
from summarization.textrank_summarizer import textrank_summary
from utils.file_handler import read_uploaded_file

# ----------------------------------------------------------------------
# Page setup
# ----------------------------------------------------------------------
st.set_page_config(page_title="News Text Summarizer", page_icon="📰", layout="wide")

EXTRACTIVE_METHODS = {
    "Lead-3 (baseline)": lead_summary,
    "Frequency-based": frequency_summary,
    "TextRank": textrank_summary,
}
ABSTRACTIVE = "Abstractive (DistilBART)"
COMPARE = "Compare all methods"
ALL_OPTIONS = [*EXTRACTIVE_METHODS, ABSTRACTIVE, COMPARE]

# ----------------------------------------------------------------------
# Styling (only affects the look of the page)
# ----------------------------------------------------------------------
CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Merriweather:wght@700;900&display=swap');

html, body, .stApp, .stMarkdown, .stTextArea textarea, button, input {
    font-family: 'Inter', 'Segoe UI', sans-serif;
}
.block-container { padding-top: 1.4rem; padding-bottom: 2rem; max-width: 1280px; }
#MainMenu, footer { visibility: hidden; }

/* ---------- Hero header ---------- */
.hero {
    position: relative; overflow: hidden;
    background: linear-gradient(120deg, #0b1f3a 0%, #133e6b 55%, #0f766e 100%);
    border-radius: 20px; padding: 30px 34px; margin-bottom: 1.4rem;
    color: #fff; box-shadow: 0 12px 32px rgba(11, 31, 58, 0.22);
}
.hero::after {
    content: "📰"; position: absolute; right: 28px; top: 4px;
    font-size: 120px; opacity: 0.12;
}
.hero h1 {
    font-family: 'Merriweather', Georgia, serif; font-weight: 900;
    font-size: 2.15rem; line-height: 1.2; margin: 0 0 8px 0; color: #fff; padding: 0;
}
.hero p { margin: 0; font-size: 1.03rem; opacity: 0.88; max-width: 760px; }
.badges { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 16px; }
.badge {
    background: rgba(255, 255, 255, 0.14); border: 1px solid rgba(255, 255, 255, 0.28);
    padding: 4px 13px; border-radius: 999px; font-size: 0.8rem; font-weight: 500;
}

/* ---------- Section titles ---------- */
.section-title {
    font-weight: 700; font-size: 1.08rem; color: #0b1f3a;
    margin: 0.3rem 0 0.6rem 0; display: flex; align-items: center; gap: 8px;
}

/* ---------- Stat cards ---------- */
.stat-grid { display: grid; grid-template-columns: repeat(5, 1fr); gap: 12px; margin: 0.4rem 0 1.2rem; }
@media (max-width: 900px) { .stat-grid { grid-template-columns: repeat(2, 1fr); } }
.stat {
    background: #fff; border: 1px solid #e3e8ef; border-radius: 14px;
    padding: 14px 16px; box-shadow: 0 2px 8px rgba(15, 23, 42, 0.04);
}
.stat .label {
    font-size: 0.72rem; font-weight: 600; letter-spacing: 0.06em;
    text-transform: uppercase; color: #64748b;
}
.stat .value { font-size: 1.6rem; font-weight: 700; color: #0b1f3a; margin-top: 2px; }
.stat.accent { background: linear-gradient(135deg, #0f766e, #14b8a6); border: none; }
.stat.accent .label, .stat.accent .value { color: #fff; }

/* ---------- Content cards ---------- */
.card {
    background: #fff; border: 1px solid #e3e8ef; border-radius: 14px;
    padding: 18px 20px; line-height: 1.75; color: #1f2937; font-size: 0.97rem;
    box-shadow: 0 2px 10px rgba(15, 23, 42, 0.05);
}
.summary-card { border-left: 5px solid #0f766e; background: #f0fdfa; font-size: 1.03rem; }
mark.hl { background: #fde68a; color: #111; padding: 1px 3px; border-radius: 4px; }
.headline {
    font-family: 'Merriweather', Georgia, serif; font-size: 1.25rem; color: #0b1f3a;
    margin: 0.2rem 0 0.8rem 0;
}
.headline span {
    font-family: 'Inter', sans-serif; font-size: 0.72rem; font-weight: 700;
    letter-spacing: 0.06em; text-transform: uppercase; color: #0f766e; display: block;
}

/* ---------- Method chips ---------- */
.chip {
    display: inline-block; font-size: 0.7rem; font-weight: 700; letter-spacing: 0.05em;
    text-transform: uppercase; padding: 3px 11px; border-radius: 999px; margin-bottom: 10px;
}
.chip-extractive { background: #e0ecff; color: #1d4ed8; }
.chip-abstractive { background: #fde7ef; color: #be185d; }
.chip-baseline { background: #eef2f7; color: #475569; }
.trophy { float: right; font-size: 0.8rem; font-weight: 600; color: #b45309; }

/* ---------- How-it-works cards ---------- */
.steps { display: grid; grid-template-columns: repeat(3, 1fr); gap: 14px; margin-top: 0.6rem; }
@media (max-width: 900px) { .steps { grid-template-columns: 1fr; } }
.step {
    background: #fff; border: 1px dashed #c7d2de; border-radius: 14px; padding: 18px;
}
.step .num {
    width: 30px; height: 30px; border-radius: 50%; background: #0f766e; color: #fff;
    display: flex; align-items: center; justify-content: center; font-weight: 700;
    margin-bottom: 8px;
}
.step b { color: #0b1f3a; }
.step p { margin: 4px 0 0 0; color: #64748b; font-size: 0.9rem; }

/* ---------- Widgets ---------- */
.stTextArea textarea { border-radius: 12px; font-size: 0.97rem; line-height: 1.6; }
.stButton > button, .stDownloadButton > button { border-radius: 10px; font-weight: 600; }
.stButton > button[kind="primary"], button[data-testid="stBaseButton-primary"] {
    background: linear-gradient(120deg, #0f766e, #14b8a6); border: none;
    box-shadow: 0 4px 14px rgba(15, 118, 110, 0.32); padding: 0.55rem 1.5rem;
}
[data-testid="stExpander"] { border-radius: 12px; }
[data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 { color: #0b1f3a; }

.app-footer { text-align: center; color: #94a3b8; font-size: 0.82rem; margin-top: 2.5rem; }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


@st.cache_resource(show_spinner=False)
def get_model():
    """Load DistilBART once per app session (Streamlit keeps it in memory)."""
    return load_model()


def md_safe(text: str) -> str:
    """Streamlit treats $...$ as maths. News often has $ amounts, so escape them."""
    return text.replace("$", "\\$")


def chip(method_name: str) -> str:
    """Coloured label showing the type of method."""
    if "Abstractive" in method_name:
        return '<span class="chip chip-abstractive">Abstractive</span>'
    if "Lead" in method_name:
        return '<span class="chip chip-baseline">Baseline</span>'
    return '<span class="chip chip-extractive">Extractive</span>'


def section(title: str) -> None:
    st.markdown(f'<div class="section-title">{title}</div>', unsafe_allow_html=True)


# ----------------------------------------------------------------------
# Session state + button callbacks
# ----------------------------------------------------------------------
if "article_text" not in st.session_state:
    st.session_state.article_text = ""
if "output" not in st.session_state:
    st.session_state.output = None
if "reference_text" not in st.session_state:
    st.session_state.reference_text = ""


def load_sample():
    name = st.session_state.get("sample_choice")
    if name:
        st.session_state.article_text = (config.SAMPLE_DIR / name).read_text(encoding="utf-8")
        st.session_state.output = None


def load_upload():
    uploaded = st.session_state.get("uploaded_file")
    if uploaded is not None:
        st.session_state.article_text = read_uploaded_file(uploaded)
        st.session_state.output = None


def clear_all():
    st.session_state.article_text = ""
    st.session_state.reference_text = ""
    st.session_state.output = None


# ----------------------------------------------------------------------
# Sidebar – settings
# ----------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Settings")
    method = st.radio("Summarization method", ALL_OPTIONS, index=2)

    st.subheader("📏 Summary length")
    num_sentences = st.slider("Sentences (extractive methods)", 1, 10,
                              config.DEFAULT_NUM_SENTENCES)
    max_len = st.slider("Max length in tokens (abstractive)", 60, 250,
                        config.ABSTRACTIVE_MAX_LENGTH, step=10)
    min_len = st.slider("Min length in tokens (abstractive)", 10, 100,
                        config.ABSTRACTIVE_MIN_LENGTH, step=5)
    if min_len >= max_len:
        min_len = max_len // 2
        st.caption(f"Min length lowered to {min_len} (must be below max).")

    st.divider()
    st.subheader("📂 Sample articles")
    samples = sorted(p.name for p in config.SAMPLE_DIR.glob("*.txt"))
    if samples:
        st.selectbox("Choose a sample", samples, key="sample_choice")
        st.button("Load sample article", on_click=load_sample, type="secondary")
    else:
        st.caption("No sample articles found in data/sample_articles/")

    st.divider()
    st.info(
        "**Extractive** methods copy the most important sentences.\n\n"
        "**Abstractive** writes a new summary with a pre-trained "
        "Transformer (DistilBART, fine-tuned on CNN/DailyMail).",
        icon="💡",
    )

# ----------------------------------------------------------------------
# Hero header
# ----------------------------------------------------------------------
st.markdown(
    """
    <div class="hero">
        <h1>Automatic Text Summarizer for News Articles</h1>
        <p>Turn long news articles into short, meaningful summaries using classical NLP
        and modern Transformer models, then compare them side by side.</p>
        <div class="badges">
            <span class="badge">NLTK preprocessing</span>
            <span class="badge">TF-IDF</span>
            <span class="badge">TextRank</span>
            <span class="badge">DistilBART</span>
            <span class="badge">ROUGE evaluation</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------
# Input card
# ----------------------------------------------------------------------
with st.container(border=True):
    section("✍️ Your news article")
    st.file_uploader("Upload a news article (.txt)", type=["txt"],
                     key="uploaded_file", on_change=load_upload)
    st.text_area("…or paste the article here", key="article_text", height=240,
                 placeholder="Paste a news article (at least 3 sentences)…")

    words_now = word_count(st.session_state.article_text)
    st.caption(f"📝 {words_now} words  ·  Method: **{method}**")

    with st.expander("📏 Optional: reference summary for ROUGE evaluation"):
        st.text_area("Paste a human-written summary of the same article. "
                     "ROUGE will compare the generated summary against it.",
                     key="reference_text", height=110)

    col_go, col_clear, _ = st.columns([2, 1, 5])
    generate = col_go.button("✨ Generate Summary", type="primary", width="stretch")
    col_clear.button("🗑️ Clear", on_click=clear_all, width="stretch")


# ----------------------------------------------------------------------
# Running the summarizers  (unchanged logic)
# ----------------------------------------------------------------------
def run(name: str, text: str):
    """Run one method. Returns (SummaryResult, seconds)."""
    if name == ABSTRACTIVE:
        with st.spinner("Loading DistilBART model (first time can take a minute)…"):
            get_model()
        with st.spinner("DistilBART is writing the summary…"):
            t0 = time.perf_counter()
            result = abstractive_summary(text, max_len, min_len)
    else:
        t0 = time.perf_counter()
        result = EXTRACTIVE_METHODS[name](text, num_sentences)
    return result, time.perf_counter() - t0


if generate:
    text = st.session_state.article_text.strip()
    if not text:
        st.warning("Please paste an article, upload a .txt file, or load a sample first.")
    else:
        names = [*EXTRACTIVE_METHODS, ABSTRACTIVE] if method == COMPARE else [method]
        results, errors = [], []
        for name in names:
            try:
                results.append(run(name, text))
            except ArticleTooShortError as e:
                errors.append(str(e))
                break                      # same problem for every method
            except Exception as e:         # e.g. model download failed
                errors.append(f"{name} failed: {e}")
        st.session_state.output = {"mode": method, "text": text,
                                   "reference": st.session_state.reference_text.strip(),
                                   "results": results, "errors": errors}


# ----------------------------------------------------------------------
# Display helpers
# ----------------------------------------------------------------------
def highlighted_article(result) -> str:
    """Article as HTML with the chosen sentences highlighted."""
    parts = []
    for s in result.sentences:
        sentence = html.escape(s.original)
        if s.index in result.selected:
            sentence = f'<mark class="hl">{sentence}</mark>'
        parts.append(sentence)
    return f'<div class="card">{" ".join(parts)}</div>'


def plain_article(text: str) -> str:
    _, body = split_title(text)
    paragraphs = [html.escape(p.strip()) for p in body.splitlines() if p.strip()]
    return '<div class="card">' + "<br><br>".join(paragraphs) + "</div>"


def show_metrics(stats: dict, seconds: float):
    cards = [
        ("Original words", stats["original_words"], ""),
        ("Summary words", stats["summary_words"], ""),
        ("Sentences", f'{stats["original_sentences"]} → {stats["summary_sentences"]}', ""),
        ("Compression", f'{stats["compression_ratio"]}%', "accent"),
        ("Time", f"{seconds:.2f} s", ""),
    ]
    items = "".join(
        f'<div class="stat {cls}"><div class="label">{label}</div>'
        f'<div class="value">{value}</div></div>'
        for label, value, cls in cards
    )
    st.html(f'<div class="stat-grid">{items}</div>')


def rouge_table(result, reference: str) -> pd.DataFrame:
    r = compute_rouge(reference, result.summary)
    return pd.DataFrame({
        "Metric": ["ROUGE-1", "ROUGE-2", "ROUGE-L"],
        "Precision": [round(r[f"{t}_p"], 3) for t in ("rouge1", "rouge2", "rougeL")],
        "Recall": [round(r[f"{t}_r"], 3) for t in ("rouge1", "rouge2", "rougeL")],
        "F1": [round(r[f"{t}_f"], 3) for t in ("rouge1", "rouge2", "rougeL")],
    })


def show_single(result, seconds: float, text: str, reference: str = ""):
    stats = compute_stats(text, result.summary)
    show_metrics(stats, seconds)

    left, right = st.columns(2, gap="large")
    with left:
        section("📄 Original article")
        with st.container(height=440, border=False):
            st.html(highlighted_article(result) if result.sentences else plain_article(text))
        if result.sentences:
            st.caption("🟨 Highlighted = sentences selected for the summary. "
                       "Junk lines and very short lines are removed before summarizing.")
    with right:
        section("✨ Generated summary")
        st.html(f'<div class="card summary-card">{chip(result.method)}'
                f'<div style="font-size:.8rem;color:#64748b;margin:-4px 0 8px">{html.escape(result.method)}</div>'
                f'{html.escape(result.summary)}</div>')
        st.download_button("⬇️ Download summary (.txt)", result.summary,
                           file_name="summary.txt", mime="text/plain")
        if reference:
            section("📏 ROUGE vs. your reference")
            st.dataframe(rouge_table(result, reference), hide_index=True,
                         width="stretch")

    if result.sentences:
        with st.expander("🔍 How were the sentences chosen? (sentence scores)"):
            df = pd.DataFrame({
                "#": [f"S{s.index + 1}" for s in result.sentences],
                "Score": result.scores,
                "Selected": ["✅" if s.index in result.selected else "" for s in result.sentences],
                "Sentence": [s.original for s in result.sentences],
            })
            st.dataframe(df, hide_index=True, width="stretch")


def show_comparison(results: list, text: str, reference: str = ""):
    rows, rouge1 = [], {}
    for result, seconds in results:
        stats = compute_stats(text, result.summary)
        row = {
            "Method": result.method,
            "Summary words": stats["summary_words"],
            "Sentences": stats["summary_sentences"],
            "Compression %": stats["compression_ratio"],
            "Time (s)": round(seconds, 2),
        }
        if reference:
            r = compute_rouge(reference, result.summary)
            rouge1[result.method] = r["rouge1_f"]
            row.update({"ROUGE-1 F1": round(r["rouge1_f"], 3),
                        "ROUGE-2 F1": round(r["rouge2_f"], 3),
                        "ROUGE-L F1": round(r["rougeL_f"], 3)})
        rows.append(row)

    best = max(rouge1, key=rouge1.get) if rouge1 else None

    section("📊 Comparison")
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")

    cols = st.columns(2, gap="large")
    for i, (result, _) in enumerate(results):
        with cols[i % 2]:
            trophy = '<span class="trophy">🏆 Best ROUGE-1</span>' if result.method == best else ""
            st.html(f'<div class="card" style="margin-bottom:14px">{trophy}{chip(result.method)}'
                    f'<div style="font-weight:700;color:#0b1f3a;margin-bottom:6px">'
                    f'{html.escape(result.method)}</div>{html.escape(result.summary)}</div>')

    with st.expander("📄 Show original article"):
        st.html(plain_article(text))


def show_welcome():
    st.html("""
    <div class="steps">
        <div class="step"><div class="num">1</div><b>Add an article</b>
            <p>Paste text, upload a .txt file, or load a sample from the sidebar.</p></div>
        <div class="step"><div class="num">2</div><b>Choose a method</b>
            <p>Lead-3, Frequency, TextRank, DistilBART, or compare all of them.</p></div>
        <div class="step"><div class="num">3</div><b>Generate</b>
            <p>See the summary, highlighted sentences, statistics and ROUGE scores.</p></div>
    </div>
    """)


# ----------------------------------------------------------------------
# Main area – output
# ----------------------------------------------------------------------
output = st.session_state.output
if output:
    st.write("")
    title, _ = split_title(output["text"])
    if title:
        st.html(f'<div class="headline"><span>Headline detected</span>{html.escape(title)}</div>')

    for err in output["errors"]:
        st.warning(err)

    if output["results"]:
        reference = output.get("reference", "")
        if output["mode"] == COMPARE:
            show_comparison(output["results"], output["text"], reference)
        else:
            result, seconds = output["results"][0]
            show_single(result, seconds, output["text"], reference)
else:
    show_welcome()

st.markdown('<div class="app-footer">Automatic Text Summarizer · NLP Mini Project · '
            'NLTK · scikit-learn · NetworkX · Hugging Face Transformers · Streamlit</div>',
            unsafe_allow_html=True)
