"""One-off CLIP retrieval script for building public-caption candidates.

This file performs model loading and dataset work at import time, so run it as
a standalone experiment rather than importing it from the training pipeline.
"""

import os
import json
import torch
import pandas as pd
from tqdm import tqdm
from transformers import CLIPTokenizer, CLIPTextModel

# ---- Load CLIP model ----
device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = CLIPTokenizer.from_pretrained(
    "clip-vit-large-patch14", local_files_only=True
)
text_model = CLIPTextModel.from_pretrained("clip-vit-large-patch14", local_files_only=True)
text_model = text_model.to(device)
text_model.eval()

# Captions must be strings; NaNs should be pre-cleaned if present.
def encode_captions(captions, batch_size=64):
    """Encode text captions into CLIP embeddings, normalized."""
    all_embs = []
    max_length = 77  # CLIP default
    with torch.no_grad():
        iter_range = range(0, len(captions), batch_size)
        iterator = tqdm(iter_range, desc="Encoding captions") if len(captions) > 100 else iter_range
        for i in iterator:
            batch = captions[i:i+batch_size]
            batch_encoding = tokenizer(batch, truncation=True, max_length=max_length, return_length=True,
                                       return_overflowing_tokens=False, padding="max_length", return_tensors="pt")
            tokens = batch_encoding["input_ids"].to(device)
            outputs = text_model(input_ids=tokens)
            emb = outputs.pooler_output
            emb = emb / emb.norm(dim=-1, keepdim=True)
            all_embs.append(emb.cpu())
    return torch.cat(all_embs, dim=0)

# ---- Load original captions from COCO-style json ----
def load_original_captions(caption_file):
    with open(caption_file, "r") as f:
        data = json.load(f)

    id_to_name = {img["id"]: os.path.basename(img["file_name"]) for img in data["images"]}
    fname_to_caps = {}
    for ann in data["annotations"]:
        fname = id_to_name.get(ann["image_id"])
        fname_to_caps.setdefault(fname, []).append(ann["caption"])
    return fname_to_caps  # dict: basename -> [captions]

# ---- Build public embeddings (average of multiple captions per image) ----
def build_pub_embeddings(df_pub, pub_caps_dict):
    pub_embs = []
    for _, row in tqdm(df_pub.iterrows(), total=len(df_pub), desc="Public embeddings"):
        fname = os.path.basename(row["file"])
        caps = pub_caps_dict.get(fname, [])
        embs = encode_captions(caps)  # (n_caps, D)
        pub_embs.append(embs.mean(dim=0, keepdim=True))
    return torch.cat(pub_embs, dim=0)

df_priv = pd.read_csv("metadata/COCO_Intern_8B_private.csv")
df_priv = df_priv[df_priv.file.str.contains("train2014")]
df_pub = pd.read_csv("metadata/MSCOCO_train_public.csv")
df_pub = df_pub[df_pub.file.str.contains("train2014")]

pub_caps_dict = load_original_captions("data/captions_train2014.json")
pub_embs = build_pub_embeddings(df_pub, pub_caps_dict)

# Encode both sets
priv_embs = encode_captions(df_priv["c1"].tolist())

# ---- Compute Top-5 public matches for each private caption ----
results = []
for i, (path_priv, cap_priv) in tqdm(
    enumerate(zip(df_priv["file"], df_priv["c1"])),
    total=len(df_priv),
    desc="Retrieving top-5 matches"
):
    # Cosine similarities because both priv_embs and pub_embs are L2-normalized
    sims = torch.mv(pub_embs, priv_embs[i])  # shape: (N_pub,)
    topk = torch.topk(sims, k=min(5, sims.numel()))
    indices = topk.indices.tolist()
    scores = [float(s) for s in topk.values.tolist()]

    # Collect top-5 public paths
    top5_paths = df_pub.iloc[indices]["file"].tolist()

    # Also record the single best for convenience
    best_pub_path = top5_paths[0]
    best_score = scores[0]

    results.append({
        "priv_path": path_priv,
        "priv_caption": cap_priv,
        "best_pub_path": best_pub_path,
        "best_similarity": best_score,
        "top5_pub_paths": " | ".join(top5_paths),
        "top5_similarities": ", ".join(f"{s:.6f}" for s in scores),
    })

df_matches = pd.DataFrame(results)
# Save to CSV for downstream usage
out_csv = "private_to_public_top5.csv"
df_matches.to_csv(out_csv, index=False)
print(f"Saved Top-5 matches to {out_csv}")
print(df_matches.head())
