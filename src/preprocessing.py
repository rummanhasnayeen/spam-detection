"""
Text preprocessing utilities.

This module:
- Cleans raw text (lowercasing, removing URLs, etc.)
- removes stopwords
- Optionally applies stemming
- Provides a helper to apply preprocessing to a whole dataset.
"""

import re
from typing import Iterable, List

import pandas as pd

# For stopword removal and stemming
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Compile regex patterns once
URL_PATTERN = re.compile(r"https?://\S+|www\.\S+")
HTML_PATTERN = re.compile(r"<.*?>")
NON_ALPHANUM_PATTERN = re.compile(r"[^a-z0-9\s]")

# Initialize stopwords and stemmer
# Make sure to run once in a Python shell:
# >>> import nltk
# >>> nltk.download("stopwords")
STOP_WORDS = set(stopwords.words("english"))
STEMMER = PorterStemmer()


def clean_text(
    text: str,
    remove_stopwords: bool = True,
    use_stemming: bool = False,
) -> str:
    """
    Clean a single text string.

    Steps:
    - Lowercase
    - Remove URLs
    - Remove HTML tags
    - Remove non-alphanumeric characters (this also removes punctuation)
    - Collapse multiple spaces
    - Remove stopwords
    - Optionally apply stemming

    Parameters
    ----------
    text : str
        Raw input text.
    remove_stopwords : bool, optional
        If True, remove English stopwords after basic cleaning. Default is True.
    use_stemming : bool, optional
        If True, apply stemming after basic cleaning (and optional stopword removal).
        Default is False.

    Returns
    -------
    str
        Cleaned text.
    """
    # Basic normalization
    text = text.lower()
    text = URL_PATTERN.sub(" ", text)
    text = HTML_PATTERN.sub(" ", text)
    text = NON_ALPHANUM_PATTERN.sub(" ", text)
    text = re.sub(r"\s+", " ", text).strip()

    # Token-level operations
    tokens = text.split()

    if remove_stopwords:
        tokens = [t for t in tokens if t not in STOP_WORDS]

    if use_stemming:
        tokens = [STEMMER.stem(t) for t in tokens]

    return " ".join(tokens)


def preprocess_texts(
    texts: Iterable[str],
    remove_stopwords: bool = True,
    use_stemming: bool = False,
) -> List[str]:
    """
    Apply cleaning to a list/iterable of texts.

    Parameters
    ----------
    texts : Iterable[str]
        Collection of raw texts.
    remove_stopwords : bool, optional
        If True, remove stopwords during cleaning. Default is True.
    use_stemming : bool, optional
        If True, apply stemming during cleaning. Default is False.

    Returns
    -------
    List[str]
        List of cleaned texts matching the original order.
    """
    return [
        clean_text(t, remove_stopwords=remove_stopwords, use_stemming=use_stemming)
        for t in texts
    ]


def preprocess_dataframe(
    df: pd.DataFrame,
    text_col: str = "text",
    remove_stopwords: bool = True,
    use_stemming: bool = False,
) -> pd.DataFrame:
    """
    Apply cleaning to the text column of a DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing a text column.
    text_col : str, optional
        Name of the text column, by default "text".
    remove_stopwords : bool, optional
        If True, remove stopwords. Default is True.
    use_stemming : bool, optional
        If True, apply stemming. Default is False.

    Returns
    -------
    pd.DataFrame
        New DataFrame with the same columns, but cleaned text column.
    """
    df = df.copy()
    df[text_col] = preprocess_texts(
        df[text_col].astype(str),
        remove_stopwords=remove_stopwords,
        use_stemming=use_stemming,
    )
    return df
