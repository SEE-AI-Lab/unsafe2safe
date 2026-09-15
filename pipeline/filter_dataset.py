"""Filter edited samples by normalized CLIP similarity."""

from argparse import ArgumentParser

import pandas as pd


def filter_by_clip_similarity(scores: pd.DataFrame, threshold: float = 0.7) -> pd.DataFrame:
    """Keep rows whose edited/original CLIP score is above ``threshold``."""
    normalized = scores["clip_edit"] / scores["clip_orig"].replace(0, pd.NA)
    filtered = scores.assign(normalized_clip=normalized)
    return filtered[filtered["normalized_clip"] > threshold].copy()


def main() -> None:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("scores_csv")
    parser.add_argument("output_csv")
    parser.add_argument("--threshold", type=float, default=0.7)
    args = parser.parse_args()

    scores = pd.read_csv(args.scores_csv)
    filtered = filter_by_clip_similarity(scores, threshold=args.threshold)
    filtered.to_csv(args.output_csv, index=False)
    print(f"Kept {len(filtered)} of {len(scores)} samples")


if __name__ == "__main__":
    main()
