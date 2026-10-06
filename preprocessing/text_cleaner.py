"""
preprocessing/text_cleaner.py
-----------------------------
MODULE 2 – TEXT PREPROCESSING

Pipeline:
    raw article
      -> clean_text()          remove HTML, URLs, junk lines, extra spaces
      -> split_sentences()     break into sentences
      -> preprocess_sentence() tokenize, lowercase, remove stopwords/punctuation, lemmatize

KEY IDEA (important for viva):
We keep TWO versions of every sentence:
    original  -> shown to the user in the summary (proper English)
    tokens    -> cleaned words used only for SCORING
"""

import re
import string
from dataclasses import dataclass, field

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import sent_tokenize, word_tokenize

import config
from utils.nltk_setup import ensure_nltk_data

# Make sure NLTK data files exist before we use them
ensure_nltk_data()

STOPWORDS = set(stopwords.words("english"))
LEMMATIZER = WordNetLemmatizer()

# Lines that often appear in copied news text but are not part of the story
JUNK_PATTERNS = [
    r"^advertisement$",
    r"^read more.*$",
    r"^also read.*$",
    r"^click here.*$",
    r"^(photo|image|file photo)\s*:.*$",
    r"^share this.*$",
    r"^follow us.*$",
]


@dataclass
class Sentence:
    """One sentence of the article, in both forms."""
    index: int                                  # position in the article (0, 1, 2, ...)
    original: str                               # exact sentence, used for display
    tokens: list = field(default_factory=list)  # processed words, used for scoring


# ----------------------------------------------------------------------
# Step 1: Clean the raw text
# ----------------------------------------------------------------------
def clean_text(text: str) -> str:
    """Remove noise from the article but keep punctuation (needed to find sentence ends)."""
    if not text:
        return ""

    text = re.sub(r"<[^>]+>", " ", text)                  # HTML tags
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)    # URLs
    text = re.sub(r"\S+@\S+\.\S+", " ", text)             # email addresses

    # Remove junk lines like "Advertisement" or "Read more: ..."
    kept_lines = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if any(re.match(p, stripped, flags=re.IGNORECASE) for p in JUNK_PATTERNS):
            continue
        # A line without ending punctuation (e.g. a heading) gets a full stop,
        # so it does not merge with the next sentence.
        if stripped[-1] not in ".!?\"'”’":
            stripped += "."
        kept_lines.append(stripped)

    text = " ".join(kept_lines)
    text = text.replace("“", '"').replace("”", '"').replace("’", "'").replace("‘", "'")
    text = re.sub(r"\s+", " ", text)                      # many spaces -> one space
    return text.strip()


# ----------------------------------------------------------------------
# Step 2: Split into sentences
# ----------------------------------------------------------------------
def split_sentences(text: str) -> list:
    """Split cleaned text into sentences using NLTK's Punkt tokenizer.
    Punkt knows that 'Dr.' or 'U.S.' are not sentence endings."""
    sentences = sent_tokenize(text)
    return [s.strip() for s in sentences if s.strip()]


# ----------------------------------------------------------------------
# Step 3: Process ONE sentence into scoring tokens
# ----------------------------------------------------------------------
def preprocess_sentence(sentence: str, remove_stopwords: bool = True,
                        lemmatize: bool = True) -> list:
    """Tokenize -> lowercase -> drop punctuation/stopwords -> lemmatize.
    Numbers are KEPT because in news they are often key facts."""
    words = word_tokenize(sentence.lower())
    tokens = []
    for w in words:
        w = w.strip(string.punctuation + "“”‘’—–")
        if not w:
            continue                                      # was only punctuation
        if remove_stopwords and w in STOPWORDS:
            continue
        if not any(ch.isalnum() for ch in w):
            continue
        if lemmatize:
            w = LEMMATIZER.lemmatize(w)
        tokens.append(w)
    return tokens


# ----------------------------------------------------------------------
# Headline detection
# ----------------------------------------------------------------------
MAX_TITLE_WORDS = 15


def split_title(text: str):
    """If the first line looks like a headline, separate it from the body.

    A line is treated as a headline when it is short (<= 15 words) and does
    NOT end with sentence punctuation, e.g. "Nashik Approves City-Wide Solar Project".
    Returns (title, body). title is "" when no headline is found.
    Why: a headline repeats the main topic words, so scoring methods love it,
    but it only repeats what the first sentence already says.
    """
    if not text:
        return "", ""
    lines = text.strip().splitlines()
    first = lines[0].strip()
    has_more_lines = len(lines) > 1
    looks_like_title = (
        has_more_lines
        and 0 < len(first.split()) <= MAX_TITLE_WORDS
        and first[-1] not in ".!?\"'”’"
    )
    if looks_like_title:
        return first, "\n".join(lines[1:])
    return "", text


# ----------------------------------------------------------------------
# Full pipeline
# ----------------------------------------------------------------------
def preprocess_article(text: str) -> list:
    """Run the whole pipeline. Returns a list of Sentence objects.
    The headline (if any) is removed, and very short sentences
    (fewer than MIN_SENTENCE_WORDS words) are skipped."""
    _, body = split_title(text)
    cleaned = clean_text(body)
    sentences = []
    for raw in split_sentences(cleaned):
        if len(raw.split()) < config.MIN_SENTENCE_WORDS:
            continue
        sentences.append(Sentence(index=len(sentences),
                                  original=raw,
                                  tokens=preprocess_sentence(raw)))
    return sentences


def word_count(text: str) -> int:
    """Count words the simple way (split on spaces)."""
    return len(text.split())


# ----------------------------------------------------------------------
# Quick demo:  python -m preprocessing.text_cleaner
# ----------------------------------------------------------------------
if __name__ == "__main__":
    sample = config.SAMPLE_DIR / "solar_project.txt"
    article = sample.read_text(encoding="utf-8")

    print("=" * 70)
    print("ORIGINAL ARTICLE")
    print("=" * 70)
    print(article)

    title, _ = split_title(article)
    sentences = preprocess_article(article)
    print("=" * 70)
    print(f"HEADLINE (kept out of the summary): {title or '(none found)'}")
    print("=" * 70)
    print(f"AFTER PREPROCESSING  ->  {len(sentences)} sentences")
    print("=" * 70)
    for s in sentences:
        print(f"\nS{s.index + 1} original : {s.original}")
        print(f"   tokens   : {s.tokens}")
