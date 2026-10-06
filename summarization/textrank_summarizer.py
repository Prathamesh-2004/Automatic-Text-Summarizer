"""
summarization/textrank_summarizer.py
------------------------------------
METHOD 2 (MAIN): TextRank – graph-based extractive summarization
(Mihalcea & Tarau, 2004 – based on Google's PageRank)

Idea: a sentence is important if it is SIMILAR to many other important sentences.

Steps:
    1. Turn every sentence into a TF-IDF vector         (scikit-learn)
    2. Similarity matrix: cosine similarity of every pair of sentences
    3. Build a graph: sentences = nodes, similarity = edge weight   (NetworkX)
    4. Run PageRank on the graph -> one score per sentence
    5. Pick the top-N sentences, keep original order
"""

import networkx as nx
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import config
from summarization.base import (SummaryResult, build_summary, prepare_sentences,
                                select_top_sentences)


def tfidf_vectors(sentences: list):
    """Step 1: each sentence -> TF-IDF vector.
    Our tokens are already cleaned, so we tell the vectorizer to just split on spaces."""
    docs = [" ".join(s.tokens) for s in sentences]
    vectorizer = TfidfVectorizer(tokenizer=str.split, token_pattern=None, lowercase=False)
    return vectorizer.fit_transform(docs)


def similarity_matrix(sentences: list) -> np.ndarray:
    """Step 2: N x N matrix, value [i][j] = how similar sentence i is to sentence j (0..1)."""
    if not any(s.tokens for s in sentences):
        return np.zeros((len(sentences), len(sentences)))
    matrix = cosine_similarity(tfidf_vectors(sentences))
    np.fill_diagonal(matrix, 0)          # a sentence should not vote for itself
    return matrix


def pagerank_scores(matrix: np.ndarray) -> list:
    """Steps 3 + 4: build the graph and run PageRank."""
    graph = nx.from_numpy_array(matrix)
    try:
        ranks = nx.pagerank(graph, alpha=0.85, weight="weight", max_iter=200)
    except nx.PowerIterationFailedConvergence:
        # Very rare. Fall back to "sum of similarities" (degree centrality).
        totals = matrix.sum(axis=1)
        ranks = {i: float(v) for i, v in enumerate(totals)}
    return [ranks[i] for i in range(len(matrix))]


def textrank_summary(text: str, num_sentences: int = config.DEFAULT_NUM_SENTENCES) -> SummaryResult:
    sentences = prepare_sentences(text)
    matrix = similarity_matrix(sentences)
    scores = [round(s, 4) for s in pagerank_scores(matrix)]
    selected = select_top_sentences(sentences, scores, num_sentences)

    return SummaryResult(
        method="TextRank",
        summary=build_summary(sentences, selected),
        selected=selected,
        scores=scores,
        sentences=sentences,
    )
