"""
config.py
---------
Central place for all project settings.
If you want to change a default (summary length, model name, etc.),
change it here instead of hunting through every file.
"""

from pathlib import Path

# ---------- Paths ----------
BASE_DIR = Path(__file__).resolve().parent          # project root folder
DATA_DIR = BASE_DIR / "data"
SAMPLE_DIR = DATA_DIR / "sample_articles"           # demo articles (.txt)
RESULTS_DIR = DATA_DIR / "results"                  # evaluation output

# ---------- Preprocessing ----------
MIN_SENTENCE_WORDS = 4      # sentences shorter than this are ignored (e.g. "Photo: PTI.")
MIN_ARTICLE_SENTENCES = 3   # article must have at least this many sentences to summarize

# ---------- Extractive summarization ----------
DEFAULT_NUM_SENTENCES = 3   # default number of sentences in an extractive summary

# ---------- Abstractive summarization (used in Phase 3) ----------
# distilbart is a smaller, faster version of bart-large-cnn (good for laptops)
ABSTRACTIVE_MODEL = "sshleifer/distilbart-cnn-12-6"
ABSTRACTIVE_MAX_LENGTH = 130   # max tokens in generated summary
ABSTRACTIVE_MIN_LENGTH = 30    # min tokens in generated summary
