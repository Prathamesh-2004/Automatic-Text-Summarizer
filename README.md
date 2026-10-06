# 📰 Automatic Text Summarizer for News Articles

An NLP mini project that automatically generates short, meaningful summaries of news articles.
It compares **classical extractive methods** (Lead-3, Frequency-based, TextRank) with a
**Transformer-based abstractive model** (DistilBART), evaluated with **ROUGE** on the
**CNN/DailyMail** benchmark, and provides a **Streamlit** web interface.

> B.Tech Computer Engineering · 3rd Year · NLP Mini Project

---

## ✨ Features

- Paste an article or upload a `.txt` file
- Four summarization methods:
  - **Lead-3** – baseline (first N sentences)
  - **Frequency-based** – scores sentences by normalized word frequency
  - **TextRank** – graph-based ranking (TF-IDF + cosine similarity + PageRank)
  - **Abstractive** – pre-trained DistilBART (`sshleifer/distilbart-cnn-12-6`)
- **Compare all methods** side by side
- Adjustable summary length
- Statistics: word count, sentence count, compression ratio, processing time
- Selected sentences **highlighted** in the original article
- **Sentence score table** showing how extractive methods decide
- Optional **ROUGE-1 / ROUGE-2 / ROUGE-L** against a reference summary
- Headline detection, junk-line removal and summary post-processing
- Download the summary as `.txt`
- Dataset evaluation script with results table and bar chart

---

## 🏗️ Architecture

```text
User ─► Input (paste / upload .txt)
          │
          ▼
   Preprocessing (NLTK)
   clean text → remove headline & junk → sentence split →
   tokenize → lowercase → remove stopwords → lemmatize
   (original sentences are kept for display)
          │
   ┌──────┴──────────────────────────┐
   ▼                                 ▼
 Extractive                      Abstractive
 • Lead-3                        • DistilBART (Hugging Face
 • Frequency                       Transformers + PyTorch)
 • TextRank (TF-IDF → cosine     • chunking for long articles
   similarity → PageRank)        • beam search
   │                                 │
   └──────────────┬──────────────────┘
                  ▼
   Post-processing (reorder, tidy punctuation)
                  ▼
   Evaluation (word counts, compression ratio, ROUGE)
                  ▼
   Streamlit UI
```

---

## 📊 Results

Evaluated on **50 articles** from the **CNN/DailyMail 3.0.0 test set**
(reference summaries average 35 words; 3-sentence extractive summaries).

| Method | ROUGE-1 F1 | ROUGE-2 F1 | ROUGE-L F1 | Avg words | Avg time (s) |
|---|---|---|---|---|---|
| **DistilBART (abstractive)** | **34.42** | **15.10** | **24.94** | 58.8 | 10.86 |
| Lead-3 (baseline) | 29.72 | 12.86 | 21.65 | 63.7 | 0.12 |
| TextRank | 28.44 | 10.22 | 19.95 | 69.5 | 0.03 |
| Frequency-based | 25.29 | 8.35 | 16.80 | 98.3 | 0.02 |

**Key findings**

1. DistilBART achieves the best scores on all ROUGE metrics.
2. Lead-3 is a strong baseline because news articles state key facts first.
3. TextRank outperforms frequency-based scoring by modelling relationships between sentences.
4. Frequency-based summaries are the longest (98 words), confirming its bias toward long sentences.
5. DistilBART is roughly 360× slower than TextRank on CPU: a clear speed vs. quality trade-off.

Full results: `data/results/rouge_summary.csv`, `rouge_per_article.csv`, `rouge_chart.png`.

---

## 🛠️ Tech Stack

| Technology | Purpose |
|---|---|
| Python 3.11 | Programming language |
| NLTK | Sentence/word tokenization, stopwords, lemmatization |
| scikit-learn | TF-IDF vectors, cosine similarity |
| NetworkX | Sentence graph + PageRank (TextRank) |
| NumPy / Pandas | Matrices and result tables |
| Hugging Face Transformers + PyTorch | Pre-trained DistilBART model |
| Hugging Face Datasets | CNN/DailyMail test articles |
| rouge-score | ROUGE evaluation |
| Matplotlib | Results chart |
| Streamlit | Web interface |

---

## 📁 Project Structure

```text
Automatic-Text-Summarizer/
├── app.py                         # Streamlit web interface
├── config.py                      # All settings in one place
├── requirements.txt
├── README.md
├── .streamlit/config.toml         # Streamlit settings
├── data/
│   ├── sample_articles/           # Demo articles (.txt)
│   ├── cnn_dailymail_test_sample.csv   # Saved evaluation sample
│   └── results/                   # ROUGE tables and chart
├── preprocessing/
│   └── text_cleaner.py            # Cleaning, headline detection, tokenization
├── summarization/
│   ├── base.py                    # Shared helpers + post-processing
│   ├── lead3_baseline.py
│   ├── frequency_summarizer.py
│   ├── textrank_summarizer.py
│   ├── abstractive_summarizer.py  # DistilBART
│   └── demo_extractive.py         # Command-line demo
├── evaluation/
│   ├── metrics.py                 # Statistics + ROUGE
│   └── evaluate_dataset.py        # CNN/DailyMail evaluation
└── utils/
    ├── nltk_setup.py              # Downloads NLTK data
    └── file_handler.py            # Reads uploaded files
```

---

## 🚀 Setup

```bash
# 1. Create and activate a virtual environment
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Mac/Linux

# 2. Install libraries
pip install -r requirements.txt

# 3. Download NLTK data (one time)
python -m utils.nltk_setup
```

The DistilBART model (~1.2 GB) downloads automatically the first time it is used.

---

## ▶️ Usage

**Web app**
```bash
streamlit run app.py
```
Opens at http://localhost:8501

**Command-line demos**
```bash
python -m preprocessing.text_cleaner                  # preprocessing steps
python -m summarization.demo_extractive mumbai_rain.txt   # 3 extractive methods
python -m summarization.abstractive_summarizer        # TextRank vs DistilBART
python -m evaluation.metrics                          # ROUGE on a textbook example
```

**Evaluation**
```bash
python -m evaluation.evaluate_dataset                 # 50 articles, all methods (~10 min)
python -m evaluation.evaluate_dataset --skip-abstractive   # extractive only (~1 min)
python -m evaluation.evaluate_dataset --n 100 --sentences 2
```

---

## ⚠️ Limitations

- Extractive summaries can feel disconnected and may contain unresolved pronouns.
- Abstractive models can occasionally generate facts not in the article (hallucination).
- DistilBART reads ~1024 tokens at a time; longer articles are split into chunks.
- DistilBART is slow on CPU (~11 s per article).
- English only.
- ROUGE measures word overlap, not meaning or factual correctness.

## 🔮 Future Scope

- Summarize directly from a news URL
- Keyword and named-entity highlighting
- Multilingual summaries (Hindi, Marathi) with mT5
- News category detection and sentiment analysis
- Fine-tuning on Indian news datasets
