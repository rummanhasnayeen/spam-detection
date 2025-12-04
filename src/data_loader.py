"""
Data loading utilities for the spam classification project.

This module is responsible for:
- Loading different datasets from disk
- Normalizing them into a common format: columns ['text', 'label'](1st dataset has v1,v2 columns)
- Converting labels into numeric form (0 = ham, 1 = spam)
"""

import pandas as pd
from typing import Tuple


def _normalize_labels(df: pd.DataFrame, label_col: str) -> pd.Series:
    """
    Convert textual labels into binary numeric labels.

    Assumes spam labels look like 'spam', 'SPAM', etc.,
    and ham labels look like 'ham', 'HAM', or anything else.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the label column.
    label_col : str
        Name of the column containing the labels.

    Returns
    -------
    pd.Series
        A series of 0/1 numeric labels.
    """
    labels = df[label_col].astype(str).str.lower().str.strip()
    return (labels == "spam").astype(int)

def load_sms_spam_dataset(path: str) -> pd.DataFrame:
    """
    Referenced as 1st dataset throughout the project
    Load the SMS spam dataset from CSV and normalize column names.

    Supports the popular 'spam.csv' format with columns:
        - v1: label (ham/spam)
        - v2: text
    as well as a custom format with:
        - class: label (ham/spam)
        - sms: text
    Returns a DataFrame with standardized columns:
        - label: 0/1 (0 = ham, 1 = spam)
        - text: message string
    """
    # Use latin-1 because the popular SMS Spam Collection is not pure UTF-8
    df = pd.read_csv(path, encoding="latin-1")

    # If the file has v1/v2 (Kaggle)
    if {"v1", "v2"}.issubset(df.columns):
        df = df.rename(columns={"v1": "class", "v2": "sms"})

    expected_cols = {"class", "sms"}
    if not expected_cols.issubset(df.columns):
        raise ValueError(
            f"Expected columns 'class' and 'sms' in {path}, "
            f"but got {list(df.columns)}"
        )

    df = df[["class", "sms"]].copy()

    # Normalize column names to match the rest of the pipeline
    df = df.rename(columns={"class": "label", "sms": "text"})

    # Map labels to binary: ham -> 0, spam -> 1
    df["label"] = df["label"].str.strip().str.lower()
    label_map = {"ham": 0, "spam": 1}
    if not set(df["label"].unique()).issubset(label_map.keys()):
        raise ValueError(
            f"Unexpected label values in SMS dataset: {df['label'].unique()}"
        )
    df["label"] = df["label"].map(label_map)

    # Drop rows with missing text just in case
    df = df.dropna(subset=["text"]).reset_index(drop=True)

    return df


# def load_email_spam_dataset(path: str = "data/email_text.csv") -> pd.DataFrame:
#     """
#     Referenced as 2nd dataset throughout the project
#     Load the email spam dataset from email_text.csv.
#
#     Expected columns:
#       - 'label': 0 (ham) or 1 (spam)
#       - 'text' : email body
#
#     Returns a DataFrame with standardized columns:
#       - 'text'
#       - 'label' (1 = spam, 0 = ham)
#     """
#     df = pd.read_csv(path)
#
#     # Ensure we only keep the two columns we need
#     df = df[["label", "text"]].copy()
#
#     # If label is 0/1, we keep as-is (0 = ham, 1 = spam)
#     df["label"] = df["label"].astype(int)
#
#     # Drop any rows with missing text or labels
#     df.dropna(subset=["text", "label"], inplace=True)
#
#     return df


def load_generic_spam_dataset(path: str, text_col: str, label_col: str) -> pd.DataFrame:
    """
    Referenced as 2nd dataset throughout the project
    Load a second spam dataset (e.g., SpamAssassin, Enron spam, etc.)
    from a CSV file and normalize it.

    Parameters
    ----------
    path : str
        Path to the CSV file.
    text_col : str
        Column name containing the message text.
    label_col : str
        Column name containing the label (spam/ham or 1/0).

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: ['text', 'label'] where label is 0/1.
    """
    df = pd.read_csv(path, encoding="utf-8")
    if text_col not in df.columns or label_col not in df.columns:
        raise ValueError(
            f"Expected columns '{text_col}' and '{label_col}' in dataset."
        )

    # If labels are not numeric, normalize them similar to SMS dataset
    if df[label_col].dtype == object:
        df["label"] = _normalize_labels(df, label_col)
    else:
        # Assume 1 = spam, 0 = ham
        df["label"] = (df[label_col] > 0).astype(int)

    df = df[[text_col, "label"]]
    df = df.rename(columns={text_col: "text"})
    df["text"] = df["text"].astype(str)
    return df
