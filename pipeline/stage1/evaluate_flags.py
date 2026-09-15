"""Evaluate Stage 1 privacy flags against VISPR annotations."""

from argparse import ArgumentParser
import csv
import json
from pathlib import Path


def evaluate_flags(prediction_csv, annotation_root, flag_column="PRIVACY_FLAG"):
    """Return binary classification metrics for a prediction CSV."""
    annotation_root = Path(annotation_root)
    true_labels = []
    predicted_labels = []

    with Path(prediction_csv).open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            image_path = Path(row["file"])
            annotation_path = annotation_root / image_path.with_suffix(".json")
            with annotation_path.open(encoding="utf-8") as annotation_handle:
                labels = json.load(annotation_handle)["labels"]
            true_labels.append(int("a0_safe" not in labels))
            predicted_labels.append(int(row[flag_column].strip().upper() == "TRUE"))

    true_positive = sum(actual == predicted == 1 for actual, predicted in zip(true_labels, predicted_labels))
    true_negative = sum(actual == predicted == 0 for actual, predicted in zip(true_labels, predicted_labels))
    false_positive = sum(actual == 0 and predicted == 1 for actual, predicted in zip(true_labels, predicted_labels))
    false_negative = sum(actual == 1 and predicted == 0 for actual, predicted in zip(true_labels, predicted_labels))
    total = len(true_labels)
    precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
    recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "accuracy": (true_positive + true_negative) / total if total else 0.0,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion_matrix": [[true_negative, false_positive], [false_negative, true_positive]],
        "count": total,
    }


def main() -> None:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("prediction_csv")
    parser.add_argument("annotation_root")
    parser.add_argument("--flag-column", default="PRIVACY_FLAG")
    args = parser.parse_args()
    metrics = evaluate_flags(args.prediction_csv, args.annotation_root, args.flag_column)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
