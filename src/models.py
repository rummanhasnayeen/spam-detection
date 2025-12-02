"""
Model definitions.

This module defines:
- Several scikit-learn classifiers
- A custom classifier implemented from scratch (SimpleCentroidClassifier)
"""

from typing import Dict, Any

import numpy as np
from scipy.sparse import csr_matrix, issparse
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

from sklearn.base import BaseEstimator, ClassifierMixin


class SimpleCentroidClassifier(BaseEstimator, ClassifierMixin):
    """
    Simple custom classifier:
    - Compute separate centroids for spam and ham in feature space.
    - Classify a sample based on which centroid it is closer to (squared Euclidean distance).
    """

    def __init__(self):
        self.spam_centroid_ = None
        self.ham_centroid_ = None
        self.classes_ = None

    def fit(self, X, y):
        """
        Fit the classifier by computing centroids of spam and ham examples.

        Parameters
        ----------
        X : scipy.sparse matrix of shape (n_samples, n_features)
            Feature matrix.
        y : array-like of shape (n_samples,)
            Labels. Assumed to be 0 for ham and 1 for spam (or 'ham' / 'spam' that
            have been mapped before calling this method).
        """
        y = np.array(y)

        # If labels are strings ('ham', 'spam'), map them to 0/1
        if y.dtype.kind in {"U", "S", "O"}:
            y_num = np.zeros_like(y, dtype=int)
            y_num[y == "spam"] = 1
            y = y_num

        self.classes_ = np.unique(y)

        # Select rows belonging to each class
        spam_mask = (y == 1)
        ham_mask = (y == 0)

        X_spam = X[spam_mask]
        X_ham = X[ham_mask]

        # Compute centroids as dense 1D arrays
        # X.mean(axis=0) returns a 1 x n_features matrix; convert to flat numpy array
        self.spam_centroid_ = np.asarray(X_spam.mean(axis=0)).ravel()
        self.ham_centroid_ = np.asarray(X_ham.mean(axis=0)).ravel()

        return self

    def predict(self, X):
        """
        Predict labels for given samples.

        Parameters
        ----------
        X : scipy.sparse matrix of shape (n_samples, n_features)

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Predicted labels (0 for ham, 1 for spam).
        """
        if self.spam_centroid_ is None or self.ham_centroid_ is None:
            raise RuntimeError("The classifier must be fitted before calling predict().")

        # Convert X to dense for distance computation
        X_dense = X.toarray()  # shape: (n_samples, n_features)

        # Compute squared Euclidean distance to each centroid
        spam_diff = X_dense - self.spam_centroid_
        ham_diff = X_dense - self.ham_centroid_

        spam_dist = np.sum(spam_diff ** 2, axis=1)
        ham_dist = np.sum(ham_diff ** 2, axis=1)

        # Closer to spam centroid => label 1, else 0
        y_pred = np.where(spam_dist < ham_dist, 1, 0)
        return y_pred


def create_classifiers() -> Dict[str, Any]:
    """
    Create the set of classifiers to be trained and compared.

    Includes:
    - Logistic Regression
    - Linear SVM
    - Multinomial Naive Bayes
    - Custom SimpleCentroidClassifier

    Returns
    -------
    Dict[str, Any]
        Mapping from model name to instantiated classifier.
    """
    classifiers = {
        "logistic_regression": LogisticRegression(
            max_iter=1000, n_jobs=-1
        ),
        "linear_svm": LinearSVC(),
        "multinomial_nb": MultinomialNB(),
        "simple_centroid": SimpleCentroidClassifier(),  # custom classifier
    }
    return classifiers
