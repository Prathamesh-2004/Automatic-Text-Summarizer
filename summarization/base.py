"""
summarization/base.py
---------------------
Shared helpers used by ALL extractive summarizers.

Every extractive summarizer follows the same 3 steps:
    1. preprocess the article into sentences      (done by preprocessing module)
    2. give every sentence a score                (different for each method)
    3. pick the top-N sentences, put them back in
       their ORIGINAL order, and join them        (same for every method -> here)
"""

import re
from dataclasses import dataclass, field

import config
from preprocessing.text_cleaner import Sentence, preprocess_article


@dataclass
class SummaryResult:
    """What every summarizer returns, so the UI can treat them all the same way."""
    method: str                                    # e.g. "TextRank"
    summary: str                                   # final summary text
    selected: list = field(default_factory=list)   # indexes of chosen sentences
    scores: list = field(default_factory=list)     # one score per sentence
    sentences: list = field(default_factory=list)  # list of Sentence objects


class ArticleTooShortError(ValueError):
    """Raised when the article has too few sentences to summarize."""


def prepare_sentences(text: str) -> list:
    """Preprocess the article and check that it is long enough."""
    sentences = preprocess_article(text)
    if len(sentences) < config.MIN_ARTICLE_SENTENCES:
        raise ArticleTooShortError(
            f"Article has only {len(sentences)} usable sentence(s). "
            f"Please provide at least {config.MIN_ARTICLE_SENTENCES}."
        )
    return sentences


def select_top_sentences(sentences: list, scores: list, num_sentences: int) -> list:
    """Return the indexes of the top-N scored sentences, sorted in ORIGINAL order.

    Why original order? A summary must read like a story. If sentence 7 scored
    higher than sentence 2, we still show sentence 2 first.
    Ties are broken in favour of the earlier sentence (news puts key facts first).
    """
    num_sentences = max(1, min(num_sentences, len(sentences)))
    ranked = sorted(range(len(sentences)), key=lambda i: (-scores[i], i))
    chosen = ranked[:num_sentences]
    return sorted(chosen)


def postprocess_summary(text: str) -> str:
    """POST-PROCESSING stage: tidy the final summary text.
    - remove spaces before punctuation  ("flooded ." -> "flooded.")
      (a known quirk of DistilBART output)
    - collapse repeated spaces
    - start with a capital letter and end with a full stop
    """
    text = re.sub(r"\s+([.,!?;:%])", r"\1", text)
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return text
    text = text[0].upper() + text[1:]
    if text[-1] not in ".!?\"'":
        text += "."
    return text


def build_summary(sentences: list, selected: list) -> str:
    """Join the ORIGINAL text of the chosen sentences."""
    return " ".join(sentences[i].original for i in selected)
