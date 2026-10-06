"""
evaluation/evaluate_dataset.py
------------------------------
Evaluates every summarization method on real news from the
CNN/DailyMail dataset (version 3.0.0, TEST split) using ROUGE.

Each dataset row has:
    article    -> full news article
    highlights -> human-written summary (3-4 bullet points)  = our REFERENCE

Usage (from the project root):
    python -m evaluation.evaluate_dataset                      (50 articles, all methods)
    python -m evaluation.evaluate_dataset --n 100              (100 articles)
    python -m evaluation.evaluate_dataset --skip-abstractive   (fast: extractive only)

Output (in data/results/):
    rouge_per_article.csv   -> every article x every method
    rouge_summary.csv       -> average scores per method   (put this in your report)
    rouge_chart.png         -> bar chart of ROUGE F1       (put this in your PPT)

Note: only the articles we need are STREAMED from Hugging Face (not the full
~1 GB dataset). They are saved to data/cnn_dailymail_test_sample.csv so the
next run works offline.
"""

import argparse
import itertools
import os
import time

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

import matplotlib

matplotlib.use("Agg")                      # draw charts to a file, no window needed
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

import config
from evaluation.metrics import compute_rouge, compute_stats
from summarization.base import ArticleTooShortError
from summarization.frequency_summarizer import frequency_summary
from summarization.lead3_baseline import lead_summary
from summarization.textrank_summarizer import textrank_summary

DATASET_NAME = "abisee/cnn_dailymail"
DATASET_VERSION = "3.0.0"
SAMPLE_CSV = config.DATA_DIR / "cnn_dailymail_test_sample.csv"


# ----------------------------------------------------------------------
# 1. Load the dataset sample
# ----------------------------------------------------------------------
def load_sample(n: int) -> pd.DataFrame:
    if SAMPLE_CSV.exists():
        df = pd.read_csv(SAMPLE_CSV)
        if len(df) >= n:
            print(f"Using saved sample: {SAMPLE_CSV.name} ({n} articles)")
            return df.head(n)

    from datasets import load_dataset
    print(f"Streaming {n} articles from {DATASET_NAME} ({DATASET_VERSION}, test split)...")
    stream = load_dataset(DATASET_NAME, DATASET_VERSION, split="test", streaming=True)
    rows = [{"id": ex["id"], "article": ex["article"], "highlights": ex["highlights"]}
            for ex in itertools.islice(stream, n)]
    df = pd.DataFrame(rows)
    SAMPLE_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(SAMPLE_CSV, index=False)
    print(f"Saved sample to {SAMPLE_CSV}")
    return df


# ----------------------------------------------------------------------
# 2. The methods to compare
# ----------------------------------------------------------------------
def build_methods(num_sentences: int, include_abstractive: bool) -> dict:
    methods = {
        f"Lead-{num_sentences}": lambda a: lead_summary(a, num_sentences).summary,
        "Frequency": lambda a: frequency_summary(a, num_sentences).summary,
        "TextRank": lambda a: textrank_summary(a, num_sentences).summary,
    }
    if include_abstractive:
        from summarization.abstractive_summarizer import abstractive_summary, load_model
        print(f"Loading {config.ABSTRACTIVE_MODEL} ...")
        load_model()                                    # load once, before timing
        methods["DistilBART"] = lambda a: abstractive_summary(a).summary
    return methods


# ----------------------------------------------------------------------
# 3. Run + score
# ----------------------------------------------------------------------
def evaluate(df: pd.DataFrame, methods: dict) -> pd.DataFrame:
    records = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Summarizing"):
        article = str(row["article"])
        reference = str(row["highlights"]).replace("\n", " ")
        for name, summarize in methods.items():
            t0 = time.perf_counter()
            try:
                summary = summarize(article)
            except ArticleTooShortError:
                continue
            seconds = time.perf_counter() - t0
            stats = compute_stats(article, summary)
            records.append({
                "article_id": row["id"],
                "method": name,
                **compute_rouge(reference, summary),
                "summary_words": stats["summary_words"],
                "compression_ratio": stats["compression_ratio"],
                "time_s": seconds,
                "summary": summary,
            })
    return pd.DataFrame(records)


def summarize_results(results: pd.DataFrame) -> pd.DataFrame:
    """Average per method. ROUGE shown as percentages (0-100)."""
    rouge_cols = [c for c in results.columns if c.startswith("rouge")]
    table = results.groupby("method", sort=False).agg(
        {**{c: "mean" for c in rouge_cols},
         "summary_words": "mean", "compression_ratio": "mean", "time_s": "mean",
         "article_id": "count"})
    table[rouge_cols] = table[rouge_cols] * 100
    table = table.rename(columns={"article_id": "articles"})
    return table.round(2)


def draw_chart(table: pd.DataFrame, path) -> None:
    metrics = [("rouge1_f", "ROUGE-1"), ("rouge2_f", "ROUGE-2"), ("rougeL_f", "ROUGE-L")]
    methods = list(table.index)
    width = 0.8 / len(methods)
    colors = ["#9aa5b1", "#4c78a8", "#2a9d8f", "#e76f51", "#8e6bbf"]

    fig, ax = plt.subplots(figsize=(9, 5))
    for i, method in enumerate(methods):
        xs = [m + i * width - 0.4 + width / 2 for m in range(len(metrics))]
        values = [table.loc[method, col] for col, _ in metrics]
        bars = ax.bar(xs, values, width, label=method, color=colors[i % len(colors)])
        ax.bar_label(bars, fmt="%.1f", fontsize=8, padding=2)

    ax.set_xticks(range(len(metrics)))
    ax.set_xticklabels([label for _, label in metrics])
    ax.set_ylabel("F1 score (%)")
    ax.set_title(f"ROUGE F1 on CNN/DailyMail test set ({int(table['articles'].max())} articles)")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="ROUGE evaluation on CNN/DailyMail")
    parser.add_argument("--n", type=int, default=50, help="number of test articles (default 50)")
    parser.add_argument("--sentences", type=int, default=config.DEFAULT_NUM_SENTENCES,
                        help="sentences per extractive summary (default 3)")
    parser.add_argument("--skip-abstractive", action="store_true",
                        help="only run the extractive methods (much faster)")
    args = parser.parse_args()

    df = load_sample(args.n)
    ref_words = df["highlights"].str.split().str.len().mean()
    methods = build_methods(args.sentences, include_abstractive=not args.skip_abstractive)

    if not args.skip_abstractive:
        print(f"Note: DistilBART takes ~5-20 s per article on CPU "
              f"(about {args.n * 12 // 60}+ minutes for {args.n} articles).")

    results = evaluate(df, methods)
    table = summarize_results(results)

    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results.to_csv(config.RESULTS_DIR / "rouge_per_article.csv", index=False)
    table.to_csv(config.RESULTS_DIR / "rouge_summary.csv")
    draw_chart(table, config.RESULTS_DIR / "rouge_chart.png")

    print("\n" + "=" * 78)
    print(f" RESULTS  –  CNN/DailyMail test set, {len(df)} articles "
          f"(reference summaries avg {ref_words:.0f} words)")
    print("=" * 78)
    f1 = table[["rouge1_f", "rouge2_f", "rougeL_f", "summary_words", "time_s"]].rename(columns={
        "rouge1_f": "ROUGE-1 F1", "rouge2_f": "ROUGE-2 F1", "rougeL_f": "ROUGE-L F1",
        "summary_words": "Avg words", "time_s": "Avg time (s)"})
    print(f1.to_string())
    print("\nSaved to data/results/: rouge_summary.csv, rouge_per_article.csv, rouge_chart.png")


if __name__ == "__main__":
    main()
