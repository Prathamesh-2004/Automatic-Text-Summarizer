"""
summarization/frequency_summarizer.py
-------------------------------------
METHOD 1: Word-Frequency based extractive summarization

Idea: words that appear many times are the topic of the article,
so sentences containing those words are important.

Steps:
    1. Count every (preprocessed) word in the whole article
    2. Normalize: weight = count / highest count   (so weights are 0..1)
    3. Sentence score = sum of the weights of its words
    4. Pick the top-N sentences, keep original order
"""

from collections import Counter

import config
from summarization.base import (SummaryResult, build_summary, prepare_sentences,
                                select_top_sentences)


def word_weights(sentences: list) -> dict:
    """Return {word: normalized frequency} for the whole article."""
    counts = Counter(word for s in sentences for word in s.tokens)
    if not counts:
        return {}
    max_count = max(counts.values())
    return {word: count / max_count for word, count in counts.items()}


def frequency_summary(text: str, num_sentences: int = config.DEFAULT_NUM_SENTENCES) -> SummaryResult:
    sentences = prepare_sentences(text)
    weights = word_weights(sentences)

    scores = [round(sum(weights.get(w, 0) for w in s.tokens), 4) for s in sentences]
    selected = select_top_sentences(sentences, scores, num_sentences)

    return SummaryResult(
        method="Frequency-based",
        summary=build_summary(sentences, selected),
        selected=selected,
        scores=scores,
        sentences=sentences,
    )
