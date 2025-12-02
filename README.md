# Spam Message Classification – Comparative Study

This project implements the experimental setup described in the LaTeX report for **automated spam message classification**, extended per instructor feedback:

- ✅ Uses **at least two datasets** (SMS + one more spam dataset)
- ✅ Trains **at least three classifiers**
- ✅ Includes a **custom classifier** implemented from scratch
- ✅ Implements **custom chi-square and mutual information** feature selection
- ✅ Performs a **comparative study** across feature-selection strategies and classifiers

---

## 1. Datasets

You need two labeled spam datasets in CSV form, each with a text column and a label column.

Suggested sources (match the LaTeX report):

1. **SMS Spam Collection** (Dataset 1 – SMS)
   - UCI Machine Learning Repository  
     (Download and convert to CSV if needed, or use Kaggle’s CSV version.)
   - After download, place the CSV as:
     - `data/sms_spam.csv`

   Accepted formats:
   - Kaggle-style: columns `v1` (label), `v2` (message text)
   - Normalized: columns `label`, `text`

2. **Second Spam Dataset** (Dataset 2 – e.g., email spam)
   - Example: a preprocessed version of the SpamAssassin corpus or Enron spam emails,
     converted to a CSV with columns:
     - `text` – message content
     - `label` – 1 or `"spam"` for spam, 0 or `"ham"` for non-spam
   - Place the file as:
     - `data/spamassassin.csv` (or a name of your choice)

The code only requires that:
- Dataset 1: can be loaded by `load_sms_spam_dataset`
- Dataset 2: can be loaded by `load_generic_spam_dataset` with the correct column names.

---

## 2. Installation

1. Create and activate a virtual environment (recommended):

```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

2. Install dependencies

```bash
pip install -r requirements.txt
```

3. Project Structure

```
spam_project/
├── data/
│   ├── sms_spam.csv
│   └── spamassassin.csv
├── src/
│   ├── data_loader.py        # Loading & normalizing datasets
│   ├── preprocessing.py      # Text cleaning (lowercase, remove URLs, etc.)
│   ├── features.py           # TF-IDF, chi-square, mutual information, feature selection
│   ├── models.py             # Classifiers (3+ from sklearn + custom classifier)
│   ├── evaluation.py         # Train/test split and metric computation
│   └── experiment.py         # Full orchestration over datasets and feature strategies
├── main.py                   # CLI entry point to run experiments
├── requirements.txt
└── README.md

```

4. How the Implementation Matches the Report

Multiple classifiers:

LogisticRegression

LinearSVC

MultinomialNB

SimpleCentroidClassifier (custom model implemented from scratch)

Custom feature selection:

compute_chi_square_scores(X, y):

Implements chi-square statistic using a 2x2 contingency table for each feature.

compute_mutual_information_scores(X, y):

Implements mutual information by computing joint and marginal probabilities for
feature presence vs. class labels.

Comparative study:

For each dataset:

Baseline: all TF-IDF features (no selection).

Feature selection: top-k features using:

Chi-square

Mutual information

For each feature-setting, each classifier is trained and evaluated.

Results are saved as a CSV with the following columns:

dataset, feature_strategy, k, classifier,
accuracy, precision, recall, f1.

5. Running the Experiments

```bash
python3 main.py \
  --sms_path data/spam.csv \
  --second_dataset_path data/email_text.csv \
  --second_text_col text \
  --second_label_col label \
  --output_csv results.csv \
  --top_k 500 1000

```

```
Explanation of arguments:

--sms_path
Path to the SMS spam CSV.

--second_dataset_path
Path to the second dataset CSV.

--second_text_col
Name of the text column in the second dataset (e.g., text).

--second_label_col
Name of the label column in the second dataset (e.g., label).

--output_csv
File where all results will be saved (default: results.csv).

--top_k
List of k values for feature selection (e.g., 500 1000).
For each k, the code will:

Select the top-k features by chi-square

Select the top-k features by mutual information

Evaluate all classifiers on each setting
```

6. Interpreting Results

The generated results.csv will contain rows like:

dataset: "sms_spam" or "second_dataset"

feature_strategy: "all", "chi_square", "mutual_information"

k: number of features used (for "all" this is the full feature count)

classifier: one of:

logistic_regression

linear_svm

multinomial_nb

simple_centroid

accuracy, precision, recall, f1: performance metrics on the test set.

You can directly use these tables and plots in your report to compare:

Datasets

Classifiers

Feature-selection methods

Different values of k

7. Task Division Example (for the report)

If you need to show equal task division among 2 teammates, you can describe it like this:

Teammate A:

Data collection and cleaning for Dataset 1 (SMS).

Implementation and testing of data_loader.py and preprocessing.py.

Implementation of one baseline classifier (e.g., Logistic Regression) and its experiments.

Teammate B:

Data collection and cleaning for Dataset 2 (email spam).

Implementation of features.py (chi-square + mutual information) and SimpleCentroidClassifier.

Full experiment orchestration (experiment.py, main.py) and result analysis.