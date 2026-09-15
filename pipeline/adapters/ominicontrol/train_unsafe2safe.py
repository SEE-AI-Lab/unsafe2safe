from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch
import yaml

from .unsafe2safe_dataset import Unsafe2SafeDataset


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
    dataset = Unsafe2SafeDataset(**dataset_config)

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
