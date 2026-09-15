from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch
import yaml

from .unsafe2safe_dataset import Unsafe2SafeDataset


def _resolve_path(config_path: Path, value: str | None) -> str | None:
    if value is None or value == "":
        return value
    path = Path(value).expanduser()
    if path.is_absolute():
        return str(path)
    for root in (Path.cwd(), config_path.parent):
        candidate = (root / path).resolve()
        if candidate.exists():
            return str(candidate)
    return str((Path.cwd() / path).resolve())


def main() -> None:
    """Load the adapter dataset and hand training to upstream OminiControl."""

    parser = argparse.ArgumentParser(description="Train OminiControl on Unsafe2Safe pairs")
    parser.add_argument(
        "--config",
        default=os.environ.get("OMINI_CONFIG", "pipeline/adapters/ominicontrol/config.example.yaml"),
    )
    args = parser.parse_args()

    config_path = Path(args.config).expanduser().resolve()
    os.environ["OMINI_CONFIG"] = str(config_path)
    with config_path.open() as handle:
        config = yaml.safe_load(handle)

    # OminiControl remains an external dependency. This import is deliberately
    # inside main so dataset/config tooling can be inspected without importing
    # the CUDA-heavy upstream training stack.
    from omini.train_flux.trainer import OminiModel, train

    train_config = config["train"]
    dataset_config = dict(train_config["dataset"])
    # Resolve existing data from the repository root first, then the config directory.
    for key in ("csv_path", "image_root", "target_root", "clip_score_path"):
        dataset_config[key] = _resolve_path(config_path, dataset_config.get(key))

    dataset = Unsafe2SafeDataset(**dataset_config)
    if torch.cuda.is_available():
        torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))

    model = OminiModel(
        flux_pipe_id=config["flux_path"],
        lora_config=train_config["lora_config"],
        device="cuda",
        dtype=getattr(torch, config["dtype"]),
        optimizer_config=train_config["optimizer"],
        model_config=config.get("model", {}),
        gradient_checkpointing=train_config.get("gradient_checkpointing", False),
    )

    # The upstream callback only samples when a test function is supplied.
    train(dataset, model, config, test_function=None)


if __name__ == "__main__":
    main()
