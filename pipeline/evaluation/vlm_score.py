"""Collect anonymization scores from VLM caption files."""

from argparse import ArgumentParser
import json
from pathlib import Path


def extract_anonymization_score(text: str):
    """Return the score from an ``ANONYMIZATION_SCORE`` line, if present."""
    for line in text.replace("**", "").splitlines():
        if line.strip().startswith("ANONYMIZATION_SCORE"):
            return float(line.split(":", 1)[1].strip())
    return None


def collect_scores(captions_dir, filename_suffix="_caption.json"):
    """Return score records from JSON caption files under a directory."""
    root = Path(captions_dir)
    records = []
    for caption_path in sorted(root.rglob(f"*{filename_suffix}")):
        with caption_path.open(encoding="utf-8") as handle:
            caption = json.load(handle)["caption"]
        score = extract_anonymization_score(caption)
        if score is not None:
            relative = caption_path.relative_to(root)
            image_name = relative.name[: -len(filename_suffix)] + ".jpg"
            records.append({"file": str(relative.with_name(image_name)), "score": score})
    return records


def main() -> None:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("captions_dir")
    parser.add_argument("output_json")
    args = parser.parse_args()
    records = collect_scores(args.captions_dir)
    with Path(args.output_json).open("w", encoding="utf-8") as handle:
        json.dump(records, handle, indent=2)
    print(f"Collected {len(records)} scores")


if __name__ == "__main__":
    main()
