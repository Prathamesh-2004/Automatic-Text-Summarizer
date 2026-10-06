"""
summarization/demo_extractive.py
--------------------------------
Runs all 3 extractive methods on a sample article and prints the results.

Usage (from the project root):
    python -m summarization.demo_extractive
    python -m summarization.demo_extractive mumbai_rain.txt
    python -m summarization.demo_extractive mumbai_rain.txt 4      (4 sentences)
"""

import sys

import config
from evaluation.metrics import compute_stats
from summarization.frequency_summarizer import frequency_summary
from summarization.lead3_baseline import lead_summary
from summarization.textrank_summarizer import textrank_summary

METHODS = [lead_summary, frequency_summary, textrank_summary]


def show(result, article: str) -> None:
    print("\n" + "=" * 75)
    print(f" {result.method}")
    print("=" * 75)

    # Sentence score table – shows HOW the method decided
    print(f"{'#':<4}{'Score':>8}  {'Chosen':<7} Sentence")
    print("-" * 75)
    for s, score in zip(result.sentences, result.scores):
        mark = "  YES" if s.index in result.selected else ""
        preview = s.original if len(s.original) <= 52 else s.original[:49] + "..."
        print(f"S{s.index + 1:<3}{score:>8.4f}  {mark:<7} {preview}")

    print("\nSUMMARY:")
    print(result.summary)

    st = compute_stats(article, result.summary)
    print(f"\nWords: {st['original_words']} -> {st['summary_words']}  |  "
          f"Sentences: {st['original_sentences']} -> {st['summary_sentences']}  |  "
          f"Compression: {st['compression_ratio']}%")


def main() -> None:
    filename = sys.argv[1] if len(sys.argv) > 1 else "solar_project.txt"
    n = int(sys.argv[2]) if len(sys.argv) > 2 else config.DEFAULT_NUM_SENTENCES

    article = (config.SAMPLE_DIR / filename).read_text(encoding="utf-8")
    print(f"Article: {filename}   |   Summary length: {n} sentences")

    for method in METHODS:
        show(method(article, n), article)


if __name__ == "__main__":
    main()
