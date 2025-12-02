"""
Text preprocessing utilities.

This module:
- Cleans raw text (lowercasing, removing URLs, etc.)
- Provides a helper to apply preprocessing to a whole dataset.
"""

import re
from typing import Iterable, List

import pandas as pd


URL_PATTERN = re.compile(r"https?://\S+|www\.\S+")
HTML_PATTERN = re.compile(r"<.*?>")
NON_ALPHANUM_PATTERN = re.compile(r"[^a-z0-9\s]")


def clean_text(text: str) -> str:
    """
    Clean a single text string.

    Steps:
    - Lowercase
    - Remove URLs
    - Remove HTML tags
    - Remove non-alphanumeric characters (keep letters, digits, spaces)

    Parameters
    ----------
    text : str
        Raw input text.

    Returns
    -------
    str
        Cleaned text.
    """
    text = text.lower()
    text = URL_PATTERN.sub(" ", text)
    text = HTML_PATTERN.sub(" ", text)
    text = NON_ALPHANUM_PATTERN.sub(" ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def preprocess_texts(texts: Iterable[str]) -> List[str]:
    """
    Apply cleaning to a list/iterable of texts.

    Parameters
    ----------
    texts : Iterable[str]
        Collection of raw texts.

    Returns
    -------
    List[str]
        List of cleaned texts matching the original order.
    """
    return [clean_text(t) for t in texts]


def preprocess_dataframe(df: pd.DataFrame, text_col: str = "text") -> pd.DataFrame:
    """
    Apply cleaning to the text column of a DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing a text column.
    text_col : str, optional
        Name of the text column, by default "text".

    Returns
    -------
    pd.DataFrame
        New DataFrame with the same columns, but cleaned text column.
    """
    df = df.copy()
    df[text_col] = preprocess_texts(df[text_col].astype(str))
    return df
