"""Build nearest public-caption candidates for an existing manifest.

This is a legacy data-preparation utility, not part of the active editor
training path. It is deliberately argument-driven so manifests do not need a
particular column naming convention.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Sequence

import pandas as pd
import torch
from tqdm import tqdm
from transformers import CLIPTextModel, CLIPTokenizer


def encode_captions(
    captions: Sequence[str],
    tokenizer: CLIPTokenizer,
    text_model: CLIPTextModel,
    device: str,
    batch_size: int = 64,
) -> torch.Tensor:
    """Encode captions as normalized CLIP pooled embeddings."""
    if not captions:
        raise ValueError("at least one caption is required")

    embeddings = []
    for start in range(0, len(captions), batch_size):
        batch = list(captions[start : start + batch_size])
        tokens = tokenizer(
            batch,
            truncation=True,
            max_length=77,
            padding="max_length",
            return_tensors="pt",
        )["input_ids"].to(device)
        with torch.no_grad():
            embedding = text_model(input_ids=tokens).pooler_output
        embeddings.append((embedding / embedding.norm(dim=-1, keepdim=True)).cpu())
    return torch.cat(embeddings, dim=0)


def load_original_captions(caption_file: str | Path) -> dict[str, list[str]]:
    """Load COCO-style annotations as ``basename -> captions``."""
    with open(caption_file) as file:
        data = json.load(file)

    id_to_name = {
        image["id"]: os.path.basename(image["file_name"])
        for image in data["images"]
    }
    captions: dict[str, list[str]] = {}
    for annotation in data["annotations"]:
        filename = id_to_name.get(annotation["image_id"])
        if filename is not None:
            captions.setdefault(filename, []).append(annotation["caption"])
    return captions


def build_public_embeddings(
    public_df: pd.DataFrame,
    public_captions: dict[str, list[str]],
    file_column: str,
    tokenizer: CLIPTokenizer,
    text_model: CLIPTextModel,
    device: str,
) -> torch.Tensor:
    """Average the available original captions for each public image."""
    embeddings = []
    for _, row in tqdm(public_df.iterrows(), total=len(public_df), desc="Public embeddings"):
        filename = os.path.basename(str(row[file_column]))
        captions = public_captions.get(filename)
        if not captions:
            raise ValueError(f"no original captions found for {filename}")
        embeddings.append(
            encode_captions(captions, tokenizer, text_model, device).mean(dim=0, keepdim=True)
        )
    if not embeddings:
        raise ValueError("the public manifest has no rows")
    return torch.cat(embeddings, dim=0)


def build_matches(args: argparse.Namespace) -> pd.DataFrame:
    """Build top-k public matches using the configured manifest columns."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = CLIPTokenizer.from_pretrained(args.model_id, local_files_only=args.local_files_only)
    text_model = CLIPTextModel.from_pretrained(args.model_id, local_files_only=args.local_files_only)
    text_model.to(device).eval()

    private_df = pd.read_csv(args.private_csv)
    public_df = pd.read_csv(args.public_csv)
    required_private = {args.private_file_column, args.private_caption_column}
    missing_private = sorted(required_private - set(private_df.columns))
    if missing_private:
        raise ValueError(f"private CSV is missing columns: {missing_private}")
    if args.public_file_column not in public_df:
        raise ValueError(f"public CSV is missing column: {args.public_file_column}")

    private_df = private_df[
        private_df[args.private_file_column].astype(str).str.contains(args.split_marker, na=False)
    ].reset_index(drop=True)
    public_df = public_df[
        public_df[args.public_file_column].astype(str).str.contains(args.split_marker, na=False)
    ].reset_index(drop=True)
    if private_df.empty or public_df.empty:
        raise ValueError("the selected split contains no private or public rows")

    public_embeddings = build_public_embeddings(
        public_df,
        load_original_captions(args.captions_json),
        args.public_file_column,
        tokenizer,
        text_model,
        device,
    )
    private_embeddings = encode_captions(
        private_df[args.private_caption_column].astype(str).tolist(),
        tokenizer,
        text_model,
        device,
    )

    rows = []
    for index, row in tqdm(private_df.iterrows(), total=len(private_df), desc="Retrieving matches"):
        similarities = torch.mv(public_embeddings, private_embeddings[index])
        topk = torch.topk(similarities, k=min(args.top_k, similarities.numel()))
        paths = public_df.iloc[topk.indices.tolist()][args.public_file_column].tolist()
        scores = [float(score) for score in topk.values.tolist()]
        rows.append(
            {
                "priv_path": row[args.private_file_column],
                "priv_caption": row[args.private_caption_column],
                "best_pub_path": paths[0],
                "best_similarity": scores[0],
                "top5_pub_paths": " | ".join(paths),
                "top5_similarities": ", ".join(f"{score:.6f}" for score in scores),
            }
        )
    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--private-csv", required=True)
    parser.add_argument("--public-csv", required=True)
    parser.add_argument("--captions-json", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--private-file-column", default="file")
    parser.add_argument("--private-caption-column", default="caption")
    parser.add_argument("--public-file-column", default="file")
    parser.add_argument("--split-marker", default="train2014")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--model-id", default="openai/clip-vit-large-patch14")
    parser.add_argument("--local-files-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.top_k < 1:
        raise ValueError("--top-k must be positive")
    matches = build_matches(args)
    output = Path(args.output_csv)
    output.parent.mkdir(parents=True, exist_ok=True)
    matches.to_csv(output, index=False)
    print(f"Saved {len(matches)} matches to {output}")


if __name__ == "__main__":
    main()
