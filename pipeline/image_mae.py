"""Load the vanilla ImageMAE model without vendoring its source code."""

from __future__ import annotations

import sys
from pathlib import Path

import torch


def import_image_mae(image_mae_root: str | Path):
    """Return the upstream model factories and position-embedding helper."""
    root = Path(image_mae_root).expanduser().resolve()
    if not (root / "models_vit.py").exists():
        raise FileNotFoundError(f"ImageMAE checkout not found: {root / 'models_vit.py'}")

    # The official repo is script-oriented, so importing it means adding its
    # checkout root to sys.path for this process.
    sys.path.insert(0, str(root))
    try:
        import models_vit
        from util.pos_embed import interpolate_pos_embed
    finally:
        sys.path.pop(0)
    return models_vit, interpolate_pos_embed


def build_classifier(
    image_mae_root: str | Path,
    model_name: str,
    num_classes: int,
    *,
    global_pool: bool = True,
    drop_path_rate: float = 0.1,
) -> tuple[torch.nn.Module, object, object]:
    """Construct a vanilla ImageMAE ViT classifier."""
    models_vit, interpolate_pos_embed = import_image_mae(image_mae_root)
    factory = getattr(models_vit, model_name)
    model = factory(
        num_classes=num_classes,
        drop_path_rate=drop_path_rate,
        global_pool=global_pool,
    )
    return model, models_vit, interpolate_pos_embed


def load_checkpoint(
    model: torch.nn.Module,
    checkpoint_path: str | Path,
    *,
    interpolate_pos_embed=None,
) -> tuple[list[str], list[str]]:
    """Load a pretraining or run checkpoint, ignoring decoder/head mismatches."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("model", checkpoint.get("state_dict", checkpoint))

    # MAE pretraining checkpoints contain decoder weights and often a 1000-way
    # head. Only keys that fit this classifier are loaded.
    cleaned = {}
    for key, value in state_dict.items():
        key = key[7:] if key.startswith("module.") else key
        if hasattr(value, "shape"):
            cleaned[key] = value
    if interpolate_pos_embed is not None and "pos_embed" in cleaned:
        interpolate_pos_embed(model, cleaned)

    model_state = model.state_dict()
    skipped = []
    for key in list(cleaned):
        if key not in model_state or cleaned[key].shape != model_state[key].shape:
            skipped.append(key)
            del cleaned[key]
    result = model.load_state_dict(cleaned, strict=False)
    return list(result.missing_keys), list(result.unexpected_keys) + skipped
