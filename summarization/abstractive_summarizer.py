"""
summarization/abstractive_summarizer.py
---------------------------------------
MODULE 5 – ABSTRACTIVE SUMMARIZATION (pre-trained Transformer)

Model: DistilBART (sshleifer/distilbart-cnn-12-6)
  - BART = a Transformer with an encoder (understands the article)
           and a decoder (writes a NEW summary word by word).
  - "cnn"  = already fine-tuned on the CNN/DailyMail news dataset.
  - "distil" = a smaller, faster copy of bart-large-cnn (good for laptops).
We do NOT train anything. This is "transfer learning": we use a model
someone else trained on ~300,000 news articles.

IMPORTANT: the model gets the CLEAN ORIGINAL text, not stopword-removed
tokens. A Transformer needs real, grammatical sentences to understand context.

The model can read at most ~1024 tokens (~700 words). Longer articles are
split into chunks of whole sentences, each chunk is summarized, and the
partial summaries are joined.
"""

import os
import time
from functools import lru_cache

# Hide the harmless Windows "symlinks" warning from Hugging Face
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

import config
from preprocessing.text_cleaner import clean_text, split_sentences, split_title
from summarization.base import ArticleTooShortError, SummaryResult, postprocess_summary

MAX_INPUT_TOKENS = 1024          # BART's input limit
CHUNK_TOKENS = 900               # leave some room below the limit


@lru_cache(maxsize=2)
def load_model(model_name: str = config.ABSTRACTIVE_MODEL):
    """Load tokenizer + model ONCE and keep them in memory.
    First run downloads ~1.2 GB into your Hugging Face cache folder."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
    model.eval()                 # inference mode (no training)
    return tokenizer, model, device


def _make_chunks(sentences: list, tokenizer) -> list:
    """Group whole sentences into chunks that fit inside the model's input limit."""
    chunks, current, current_len = [], [], 0
    for sent in sentences:
        n_tokens = len(tokenizer.tokenize(sent))
        if current and current_len + n_tokens > CHUNK_TOKENS:
            chunks.append(" ".join(current))
            current, current_len = [], 0
        current.append(sent)
        current_len += n_tokens
    if current:
        chunks.append(" ".join(current))
    return chunks


def _generate(text: str, tokenizer, model, device, max_length: int, min_length: int) -> str:
    inputs = tokenizer(text, return_tensors="pt", truncation=True,
                       max_length=MAX_INPUT_TOKENS).to(device)
    with torch.no_grad():        # no gradients needed -> faster, less memory
        output_ids = model.generate(
            **inputs,
            num_beams=4,               # beam search: keep 4 best candidates at each step
            max_length=max_length,
            min_length=min_length,
            no_repeat_ngram_size=3,    # never repeat the same 3 words
            length_penalty=2.0,        # >1 encourages slightly longer summaries
            early_stopping=True,
        )
    return tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()


def abstractive_summary(text: str,
                        max_length: int = config.ABSTRACTIVE_MAX_LENGTH,
                        min_length: int = config.ABSTRACTIVE_MIN_LENGTH) -> SummaryResult:
    _, body = split_title(text)
    cleaned = clean_text(body)
    sentences = [s for s in split_sentences(cleaned)
                 if len(s.split()) >= config.MIN_SENTENCE_WORDS]
    if len(sentences) < config.MIN_ARTICLE_SENTENCES:
        raise ArticleTooShortError(
            f"Article has only {len(sentences)} usable sentence(s). "
            f"Please provide at least {config.MIN_ARTICLE_SENTENCES}."
        )

    tokenizer, model, device = load_model()
    chunks = _make_chunks(sentences, tokenizer)

    if len(chunks) == 1:
        summary = _generate(chunks[0], tokenizer, model, device, max_length, min_length)
    else:
        # Long article: summarize each chunk with a shorter length, then join.
        per_chunk_max = max(40, max_length // len(chunks) + 20)
        per_chunk_min = min(min_length, per_chunk_max // 2)
        parts = [_generate(c, tokenizer, model, device, per_chunk_max, per_chunk_min)
                 for c in chunks]
        summary = " ".join(parts)

    return SummaryResult(
        method=f"Abstractive ({config.ABSTRACTIVE_MODEL.split('/')[-1]})",
        summary=postprocess_summary(summary),
    )


# ----------------------------------------------------------------------
# Quick demo:  python -m summarization.abstractive_summarizer [file.txt]
# ----------------------------------------------------------------------
if __name__ == "__main__":
    import sys

    from evaluation.metrics import compute_stats
    from summarization.textrank_summarizer import textrank_summary

    filename = sys.argv[1] if len(sys.argv) > 1 else "mumbai_rain.txt"
    article = (config.SAMPLE_DIR / filename).read_text(encoding="utf-8")

    print(f"Article: {filename}")
    print(f"Loading model '{config.ABSTRACTIVE_MODEL}' "
          "(first time downloads ~1.2 GB, please wait)...")
    t0 = time.time()
    _, _, device = load_model()
    print(f"Model loaded on {device.upper()} in {time.time() - t0:.1f} s\n")

    t0 = time.time()
    abs_result = abstractive_summary(article)
    abs_time = time.time() - t0

    t0 = time.time()
    tr_result = textrank_summary(article)
    tr_time = time.time() - t0

    for result, seconds in [(tr_result, tr_time), (abs_result, abs_time)]:
        st = compute_stats(article, result.summary)
        print("=" * 75)
        print(f" {result.method}   ({seconds:.2f} s)")
        print("=" * 75)
        print(result.summary)
        print(f"\nWords: {st['original_words']} -> {st['summary_words']}  |  "
              f"Sentences: {st['original_sentences']} -> {st['summary_sentences']}  |  "
              f"Compression: {st['compression_ratio']}%\n")
