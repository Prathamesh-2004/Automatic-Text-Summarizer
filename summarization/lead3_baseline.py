"""
summarization/lead3_baseline.py
-------------------------------
BASELINE: Lead-N (usually Lead-3)

Idea: just take the first N sentences of the article.
Journalists write the most important facts first ("inverted pyramid"),
so this simple trick is surprisingly strong for news.
Every news summarizer should be compared against it.
"""

import config
from summarization.base import SummaryResult, build_summary, prepare_sentences


def lead_summary(text: str, num_sentences: int = config.DEFAULT_NUM_SENTENCES) -> SummaryResult:
    sentences = prepare_sentences(text)
    n = max(1, min(num_sentences, len(sentences)))

    # Score = position-based: earlier sentence -> higher score (only for display)
    scores = [round(1 - i / len(sentences), 4) for i in range(len(sentences))]
    selected = list(range(n))

    return SummaryResult(
        method=f"Lead-{n} (baseline)",
        summary=build_summary(sentences, selected),
        selected=selected,
        scores=scores,
        sentences=sentences,
    )
