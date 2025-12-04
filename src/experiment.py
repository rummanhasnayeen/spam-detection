"""
Experiment orchestration.

This module:
- Ties together data loading, preprocessing, feature extraction,
  feature selection (chi-square, mutual information), and evaluation.
- Runs a comparative study across:
    * multiple classifiers
    * multiple feature selection strategies
    * multiple datasets
"""
from __future__ import annotations

from typing import Dict, Any, List

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

from .data_loader import load_sms_spam_dataset, load_generic_spam_dataset
from .preprocessing import preprocess_dataframe
from .features import (
    TfidfFeaturizer,
    compute_chi_square_scores,
    compute_mutual_information_scores,
    select_top_k_features,
)
from .models import create_classifiers
from .evaluation import train_test_split_texts, evaluate_classifier


def _run_single_dataset_experiments(
        df: pd.DataFrame,
        dataset_name: str,
        top_k_list: List[int],
        use_stemming: bool = False,
) -> List[Dict[str, Any]]:
    """
    Run all experiments for a single dataset.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset with columns ['text', 'label'].
    dataset_name : str
        Name of the dataset (for reporting).
    top_k_list : List[int]
        List of k values for feature selection.
    stemming : bool, optional

    Returns
    -------
    List[Dict[str, Any]]
        List of result dictionaries for each experiment setting.
    """
    df = preprocess_dataframe(df, text_col="text", use_stemming=use_stemming)

    texts = df["text"].values
    labels = df["label"].values.astype(int)

    X_train_texts, X_test_texts, y_train, y_test = train_test_split_texts(texts, labels)

    featurizer = TfidfFeaturizer()
    X_train = featurizer.fit_transform(list(X_train_texts))
    X_test = featurizer.transform(list(X_test_texts))

    feature_names = featurizer.get_feature_names()

    # Compute feature scores on training data
    chi2_scores = compute_chi_square_scores(X_train, y_train)
    mi_scores = compute_mutual_information_scores(X_train, y_train)

    classifiers = create_classifiers()
    results: List[Dict[str, Any]] = []

    # Baseline: all features (no feature selection)
    for clf_name, clf in classifiers.items():
        metrics = evaluate_classifier(clf, X_train, y_train, X_test, y_test)
        results.append(
            {
                "dataset": dataset_name,
                "feature_strategy": "all",
                "k": len(feature_names),
                "classifier": clf_name,
                **metrics,
            }
        )

    # Feature selection variants
    for k in top_k_list:
        # Chi-square
        X_train_chi2, chi2_idx = select_top_k_features(X_train, chi2_scores, k)
        X_test_chi2 = X_test[:, chi2_idx]

        for clf_name, clf in create_classifiers().items():
            metrics = evaluate_classifier(clf, X_train_chi2, y_train, X_test_chi2, y_test)
            results.append(
                {
                    "dataset": dataset_name,
                    "feature_strategy": "chi_square",
                    "k": int(X_train_chi2.shape[1]),
                    "classifier": clf_name,
                    **metrics,
                }
            )

        # Mutual information
        X_train_mi, mi_idx = select_top_k_features(X_train, mi_scores, k)
        X_test_mi = X_test[:, mi_idx]

        for clf_name, clf in create_classifiers().items():
            metrics = evaluate_classifier(clf, X_train_mi, y_train, X_test_mi, y_test)
            results.append(
                {
                    "dataset": dataset_name,
                    "feature_strategy": "mutual_information",
                    "k": int(X_train_mi.shape[1]),
                    "classifier": clf_name,
                    **metrics,
                }
            )

    return results


def run_dataset_1_experiment(
        sms_path: str,
        top_k_list: List[int] | None = None,
        stemming:bool = False
) -> pd.DataFrame:
    """
    Run the experiment across first dataset.

    Parameters
    ----------
    sms_path : str
        Path to the SMS spam dataset CSV file.
    top_k_list : List[int] | None, optional
        List of k values for feature selection.
        If None, defaults to [500, 1000].
    stemming : bool, optional

    Returns
    -------
    pd.DataFrame
        DataFrame summarizing experiment results.
    """
    if sms_path is None:
        sms_path = "data\spam.csv"

    if top_k_list is None:
        top_k_list = [500, 1000]

    # Dataset 1: SMS spam
    sms_df = load_sms_spam_dataset(sms_path)
    results_sms = _run_single_dataset_experiments(
        sms_df, dataset_name="sms_spam", top_k_list=top_k_list, use_stemming=stemming
    )

    return pd.DataFrame(results_sms)


def run_dataset_2_experiment(
        second_dataset_path: str,
        top_k_list: List[int] | None = None,
        stemming:bool = False
) -> pd.DataFrame:
    """
    Run the experiment across second dataset.

    Parameters
    ----------
    second_dataset_path : str
        Path to the second spam dataset CSV file.
    top_k_list : List[int] | None, optional
        List of k values for feature selection.
        If None, defaults to [500, 1000].
    stemming : bool, optional

    Returns
    -------
    pd.DataFrame
        DataFrame summarizing experiment results.
    """
    if top_k_list is None:
        top_k_list = [500, 1000]

    if second_dataset_path is None:
        second_dataset_path = "data\email_text.csv"

    second_text_col = "text"
    second_label_col = "label"

    # Dataset 2: generic (Enron)
    ds2_df = load_generic_spam_dataset(
        second_dataset_path, text_col=second_text_col, label_col=second_label_col
    )
    results_ds2 = _run_single_dataset_experiments(
        ds2_df, dataset_name="second_dataset", top_k_list=top_k_list, use_stemming=stemming
    )

    return pd.DataFrame(results_ds2)
