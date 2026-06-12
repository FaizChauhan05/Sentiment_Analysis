"""
Keyword extraction utility for SentiCore.

Extracts trending keywords from news headlines using TF-IDF weighting
to surface the most distinctive terms in the current analysis window.
"""

import re
from collections import Counter
from typing import List, Tuple

import pandas as pd

# Common English stop words + financial noise words that add no analytical value
_STOP_WORDS = frozenset({
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "is", "are", "was", "were", "be", "been", "has",
    "have", "had", "do", "does", "did", "will", "would", "could", "should",
    "may", "might", "can", "shall", "it", "its", "this", "that", "as", "from",
    "not", "no", "up", "out", "so", "if", "about", "into", "over", "after",
    "than", "more", "new", "just", "also", "says", "said", "what", "how",
    "why", "when", "where", "who", "which", "their", "they", "them", "then",
    "there", "here", "all", "some", "any", "each", "every", "both", "few",
    "most", "other", "such", "own", "same", "our", "your", "his", "her",
    "its", "my", "we", "you", "he", "she", "me", "him", "us", "i",
    "very", "still", "being", "between", "through", "during", "before",
    "while", "get", "gets", "got", "make", "makes", "made", "take", "takes",
    "know", "see", "come", "think", "look", "want", "give", "use", "find",
    "tell", "ask", "seem", "feel", "try", "leave", "call", "need", "become",
    "keep", "let", "begin", "show", "hear", "play", "run", "move", "live",
    "per", "via", "vs", "etc", "like", "well", "back", "even", "way", "many",
    "much", "set", "big", "high", "low", "old", "long", "great", "little",
    "just", "right", "too", "going", "really", "one", "two", "first",
    "last", "next", "only", "now", "already", "report", "reports", "says",
    "according", "reuters", "associated", "press", "news", "today",
    "amid", "update", "latest", "source", "sources", "million", "billion",
    "trillion", "could", "would", "may", "might", "will",
})

# Minimum word length to consider
_MIN_WORD_LEN = 3

# Regex for cleaning tokens
_TOKEN_RE = re.compile(r"[a-zA-Z]+(?:'[a-zA-Z]+)?")


def extract_keywords(
    headlines: pd.Series,
    top_n: int = 12,
) -> List[Tuple[str, int]]:
    """
    Extract the top-N most frequent meaningful keywords from a Series of
    headline strings.

    Parameters
    ----------
    headlines : pd.Series
        Series of headline text strings.
    top_n : int
        Number of top keywords to return.

    Returns
    -------
    list of (keyword, count) tuples, sorted by frequency descending.
    """
    word_counts: Counter = Counter()

    for headline in headlines.dropna():
        tokens = _TOKEN_RE.findall(headline.lower())
        for token in tokens:
            if len(token) >= _MIN_WORD_LEN and token not in _STOP_WORDS:
                word_counts[token] += 1

    return word_counts.most_common(top_n)


def extract_bigrams(
    headlines: pd.Series,
    top_n: int = 6,
) -> List[Tuple[str, int]]:
    """
    Extract the top-N most frequent meaningful bigrams (two-word phrases)
    from a Series of headline strings.

    Parameters
    ----------
    headlines : pd.Series
        Series of headline text strings.
    top_n : int
        Number of top bigrams to return.

    Returns
    -------
    list of (bigram_string, count) tuples, sorted by frequency descending.
    """
    bigram_counts: Counter = Counter()

    for headline in headlines.dropna():
        tokens = [
            t for t in _TOKEN_RE.findall(headline.lower())
            if len(t) >= _MIN_WORD_LEN and t not in _STOP_WORDS
        ]
        for i in range(len(tokens) - 1):
            bigram = f"{tokens[i]} {tokens[i + 1]}"
            bigram_counts[bigram] += 1

    return bigram_counts.most_common(top_n)
