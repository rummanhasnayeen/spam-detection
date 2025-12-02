"""
Evaluation utilities.

This module:
- Splits data into train/test sets
- Trains and evaluates models
- Computes accuracy, precision, recall, F1
"""

from typing import Dict, Any, Tuple

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split


def train_test_split_texts(
    texts: np.ndarray,
    labels: np.ndarray,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split raw texts and labels into train/test subsets.

    Parameters
    ----------
    texts : np.ndarray
        Array of raw texts.
    labels : np.ndarray
        Array of labels (0/1).
    test_size : float, optional
        Fraction of data used for test set, by default 0.2.
    random_state : int, optional
        Random seed, by default 42.

    Returns
    -------
    (np.ndarray, np.ndarray, np.ndarray, np.ndarray)
        X_train_texts, X_test_texts, y_train, y_test
    """
    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=test_size, random_state=random_state, stratify=labels
    )
    return X_train, X_test, y_train, y_test


def evaluate_classifier(
    clf: Any, X_train: csr_matrix, y_train: np.ndarray, X_test: csr_matrix, y_test: np.ndarray
) -> Dict[str, float]:
    """
    Fit a classifier and compute evaluation metrics.

    Parameters
    ----------
    clf : Any
        Classifier object with fit() and predict().
    X_train : csr_matrix
        Training feature matrix.
    y_train : np.ndarray
        Training labels.
    X_test : csr_matrix
        Test feature matrix.
    y_test : np.ndarray
        Test labels.

    Returns
    -------
    Dict[str, float]
        Dictionary of metrics: accuracy, precision, recall, f1.
    """
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
    }
    return metrics
