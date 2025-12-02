"""
Feature extraction and feature selection utilities.

This module:
- Transforms text into vector representations (TF-IDF)
- Implements custom chi-square and mutual information
  feature selection from scratch (no sklearn.feature_selection).
"""
from __future__ import annotations

from typing import Tuple, List

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer


class TfidfFeaturizer:
    """
    TF-IDF feature extractor wrapper.

    Uses scikit-learn's TfidfVectorizer internally.
    """

    def __init__(
        self,
        ngram_range=(1, 2),
        min_df: int = 2,
        max_df: float = 0.95,
        max_features: int | None = None,
    ) -> None:
        """
        Initialize the featurizer with configuration.

        Parameters
        ----------
        ngram_range : tuple, optional
            N-gram range, by default (1, 2).
        min_df : int, optional
            Minimum document frequency, by default 2.
        max_df : float, optional
            Maximum document frequency proportion, by default 0.95.
        max_features : int | None, optional
            Maximum number of features to keep, by default None.
        """
        self.vectorizer = TfidfVectorizer(
            ngram_range=ngram_range,
            min_df=min_df,
            max_df=max_df,
            max_features=max_features,
        )

    def fit_transform(self, texts: List[str]) -> csr_matrix:
        """
        Fit the vectorizer on training texts and transform them.

        Parameters
        ----------
        texts : List[str]
            Training texts.

        Returns
        -------
        csr_matrix
            TF-IDF feature matrix.
        """
        return self.vectorizer.fit_transform(texts)

    def transform(self, texts: List[str]) -> csr_matrix:
        """
        Transform new texts using the fitted vectorizer.

        Parameters
        ----------
        texts : List[str]
            New texts.

        Returns
        -------
        csr_matrix
            TF-IDF feature matrix.
        """
        return self.vectorizer.transform(texts)

    def get_feature_names(self) -> List[str]:
        """
        Get the list of feature names.

        Returns
        -------
        List[str]
            List of feature names.
        """
        return list(self.vectorizer.get_feature_names_out())


def _binary_presence(X: csr_matrix) -> csr_matrix:
    """
    Convert a count/TF-IDF matrix into a binary presence/absence matrix.

    Parameters
    ----------
    X : csr_matrix
        Input sparse matrix.

    Returns
    -------
    csr_matrix
        Matrix with the same structure but values in {0,1}.
    """
    X_bin = X.copy()
    X_bin.data = np.ones_like(X_bin.data)
    return X_bin


def compute_chi_square_scores(X: csr_matrix, y: np.ndarray) -> np.ndarray:
    """
    Compute chi-square scores for each feature.

    This is a custom implementation (no sklearn.feature_selection.chi2).

    For each feature j, we build a 2x2 contingency table:

        Feature present    Feature absent
    --------------------------------------
    Class = spam (1):  A              C
    Class = ham  (0):  B              D

    Then:

        chi2_j = N * (A*D - B*C)^2 / ((A+C)*(B+D)*(A+B)*(C+D))

    Parameters
    ----------
    X : csr_matrix
        Document-term matrix (preferably binary or TF-IDF).
    y : np.ndarray
        Binary labels, shape (n_samples,).

    Returns
    -------
    np.ndarray
        Chi-square scores, shape (n_features,).
    """
    if not isinstance(X, csr_matrix):
        X = csr_matrix(X)

    X_bin = _binary_presence(X)
    N = X_bin.shape[0]

    y = np.asarray(y).astype(int)
    if not set(np.unique(y)).issubset({0, 1}):
        raise ValueError("Labels y must be binary {0,1} for chi-square computation.")

    # Masks for classes
    pos_mask = y == 1
    neg_mask = y == 0

    # A: feature present & class=1
    A = X_bin[pos_mask].sum(axis=0)  # shape (1, n_features)
    # B: feature present & class=0
    B = X_bin[neg_mask].sum(axis=0)

    A = np.asarray(A).flatten()
    B = np.asarray(B).flatten()

    n_pos = pos_mask.sum()
    n_neg = neg_mask.sum()

    # C: feature absent & class=1
    C = n_pos - A
    # D: feature absent & class=0
    D = n_neg - B

    # Avoid division-by-zero
    eps = 1e-12
    numerator = (A * D - B * C) ** 2 * N
    denominator = (A + C) * (B + D) * (A + B) * (C + D) + eps
    chi2_scores = numerator / denominator
    return chi2_scores


def compute_mutual_information_scores(X: csr_matrix, y: np.ndarray) -> np.ndarray:
    """
    Compute mutual information scores for each feature.

    This is a custom implementation (no sklearn.feature_selection.mutual_info_classif).

    We consider binary feature presence X in {0,1} and class Y in {0,1}.

    For each feature j, using counts A, B, C, D as in chi-square, we compute:

        MI(X;Y) = sum_{x in {0,1}} sum_{y in {0,1}} p(x,y) * log( p(x,y) / (p(x)*p(y)) )

    Parameters
    ----------
    X : csr_matrix
        Document-term matrix.
    y : np.ndarray
        Binary labels, shape (n_samples,).

    Returns
    -------
    np.ndarray
        Mutual information scores, shape (n_features,).
    """
    if not isinstance(X, csr_matrix):
        X = csr_matrix(X)

    X_bin = _binary_presence(X)
    N = X_bin.shape[0]

    y = np.asarray(y).astype(int)
    if not set(np.unique(y)).issubset({0, 1}):
        raise ValueError("Labels y must be binary {0,1} for mutual information.")

    # Masks for classes
    pos_mask = y == 1
    neg_mask = y == 0

    # A: feature present & class=1
    A = X_bin[pos_mask].sum(axis=0)
    # B: feature present & class=0
    B = X_bin[neg_mask].sum(axis=0)

    A = np.asarray(A).flatten().astype(float)
    B = np.asarray(B).flatten().astype(float)

    n_pos = float(pos_mask.sum())
    n_neg = float(neg_mask.sum())
    N = float(N)

    # C: feature absent & class=1
    C = n_pos - A
    # D: feature absent & class=0
    D = n_neg - B

    # Joint probabilities
    p_x1_y1 = A / N
    p_x1_y0 = B / N
    p_x0_y1 = C / N
    p_x0_y0 = D / N

    # Marginal probabilities
    p_x1 = (A + B) / N
    p_x0 = (C + D) / N
    p_y1 = (A + C) / N
    p_y0 = (B + D) / N

    eps = 1e-12
    mi = np.zeros_like(A)

    def safe_term(p_xy, p_x, p_y):
        mask = p_xy > 0
        return np.where(
            mask,
            p_xy * np.log((p_xy + eps) / ((p_x * p_y) + eps)),
            0.0,
        )

    mi += safe_term(p_x1_y1, p_x1, p_y1)
    mi += safe_term(p_x1_y0, p_x1, p_y0)
    mi += safe_term(p_x0_y1, p_x0, p_y1)
    mi += safe_term(p_x0_y0, p_x0, p_y0)

    return mi


def select_top_k_features(
    X: csr_matrix, scores: np.ndarray, k: int
) -> Tuple[csr_matrix, np.ndarray]:
    """
    Select the top-k features given a score vector.

    Parameters
    ----------
    X : csr_matrix
        Original feature matrix.
    scores : np.ndarray
        Feature scores, shape (n_features,).
    k : int
        Number of features to keep.

    Returns
    -------
    (csr_matrix, np.ndarray)
        - Reduced feature matrix with k selected columns.
        - Indices of selected features (sorted by score descending).
    """
    k = min(k, X.shape[1])
    idx_sorted = np.argsort(scores)[::-1]
    selected_indices = idx_sorted[:k]
    X_reduced = X[:, selected_indices]
    return X_reduced, selected_indices
