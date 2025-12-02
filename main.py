"""
Entry point script.

This script:
- Parses command-line arguments
- Runs the full experimental pipeline
- Saves results to a CSV file

Usage example (after setting up datasets):

    python main.py \
        --sms_path data/sms_spam.csv \
        --second_dataset_path data/spamassassin.csv \
        --second_text_col text \
        --second_label_col label \
        --output_csv results.csv
"""

import argparse
from pathlib import Path

from src.experiment import run_full_experiment


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for running experiments.

    Returns
    -------
    argparse.Namespace
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Spam classification comparative study")

    parser.add_argument(
        "--sms_path",
        type=str,
        required=True,
        help="Path to SMS spam CSV dataset (e.g., UCI SMS Spam Collection).",
    )
    parser.add_argument(
        "--second_dataset_path",
        type=str,
        required=True,
        help="Path to second spam dataset CSV file.",
    )
    parser.add_argument(
        "--second_text_col",
        type=str,
        required=True,
        help="Name of the text column in the second dataset CSV.",
    )
    parser.add_argument(
        "--second_label_col",
        type=str,
        required=True,
        help="Name of the label column in the second dataset CSV.",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default="results.csv",
        help="Path to save the experiment results as CSV.",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        nargs="*",
        default=[500, 1000],
        help="List of k values for feature selection (e.g. --top_k 500 1000).",
    )

    return parser.parse_args()


def main() -> None:
    """
    Run the full experiment and save the results.
    """
    args = parse_args()

    results_df = run_full_experiment(
        sms_path=args.sms_path,
        second_dataset_path=args.second_dataset_path,
        second_text_col=args.second_text_col,
        second_label_col=args.second_label_col,
        top_k_list=args.top_k,
    )

    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_path, index=False)
    print(f"Saved results to {output_path}")


if __name__ == "__main__":
    main()
