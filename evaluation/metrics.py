"""
evaluation/metrics.py
---------------------
MODULE 6 – SUMMARY EVALUATION

1. compute_stats()  -> word counts, sentence counts, compression ratio
2. compute_rouge()  -> ROUGE-1, ROUGE-2, ROUGE-L (precision, recall, F1)

ROUGE compares OUR summary with a HUMAN-written reference summary:
    ROUGE-1 : overlapping single words
    ROUGE-2 : overlapping word pairs
    ROUGE-L : longest common subsequence (same order, gaps allowed)
    Recall    = overlap / words in reference   (how much did we cover?)
    Precision = overlap / words in our summary (how much of ours is relevant?)
    F1        = balance of precision and recall  (the number we report)

Quick check with the textbook example:   python -m evaluation.metrics
"""

from nltk.tokenize import sent_tokenize
from rouge_score import rouge_scorer

from preprocessing.text_cleaner import clean_text, split_title, word_count

ROUGE_TYPES = ["rouge1", "rouge2", "rougeL"]
# use_stemmer=True -> "rains" and "rain" count as the same word
_SCORER = rouge_scorer.RougeScorer(ROUGE_TYPES, use_stemmer=True)


def compute_stats(original: str, summary: str) -> dict:
    """Word/sentence counts and compression ratio (summary words / original words).
    The headline is not counted, so the numbers match what the summarizers see."""
    _, body = split_title(original)
    cleaned = clean_text(body)
    orig_words = word_count(cleaned)
    summ_words = word_count(summary)
    return {
        "original_words": orig_words,
        "summary_words": summ_words,
        "original_sentences": len(sent_tokenize(cleaned)) if cleaned else 0,
        "summary_sentences": len(sent_tokenize(summary)) if summary else 0,
        "compression_ratio": round(100 * summ_words / orig_words, 1) if orig_words else 0.0,
    }


def compute_rouge(reference: str, summary: str) -> dict:
    """Returns a flat dict, values between 0 and 1:
    {'rouge1_p':.., 'rouge1_r':.., 'rouge1_f':.., 'rouge2_p':.., ..., 'rougeL_f':..}"""
    scores = _SCORER.score(reference, summary)       # order: (target, prediction)
    result = {}
    for rtype in ROUGE_TYPES:
        s = scores[rtype]
        result[f"{rtype}_p"] = s.precision
        result[f"{rtype}_r"] = s.recall
        result[f"{rtype}_f"] = s.fmeasure
    return result


if __name__ == "__main__":
    reference = "the cat sat on the mat"
    summary = "the cat was on the mat"
    print(f"Reference : {reference}")
    print(f"Summary   : {summary}\n")
    r = compute_rouge(reference, summary)
    print(f"{'Metric':<9}{'Precision':>10}{'Recall':>9}{'F1':>8}")
    for rtype, label in [("rouge1", "ROUGE-1"), ("rouge2", "ROUGE-2"), ("rougeL", "ROUGE-L")]:
        print(f"{label:<9}{r[rtype + '_p']:>10.3f}{r[rtype + '_r']:>9.3f}{r[rtype + '_f']:>8.3f}")
    print("\nExpected: ROUGE-1 = 0.833, ROUGE-2 = 0.600, ROUGE-L = 0.833")
