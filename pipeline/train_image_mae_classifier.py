"""Train a downstream classifier with the vanilla ImageMAE ViT encoder."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from pipeline.image_mae import build_classifier, load_checkpoint
from pipeline.mae_dataset import (
    ImageMAEClassificationDataset,
    build_class_mapping,
    build_transform,
    read_manifest,
    split_manifest,
)


def get_args_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-mae-root", required=True, help="Vanilla facebookresearch/mae checkout")
    parser.add_argument("--metadata", required=True, help="CSV containing file, class, and split columns")
    parser.add_argument("--image-root", required=True, help="Root containing original images")
    parser.add_argument("--generated-root", default=None, help="Root containing edited images for unsafe rows")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--finetune", default=None, help="MAE pretraining or classifier checkpoint")
    parser.add_argument("--resume", default=None, help="Unsafe2Safe ImageMAE checkpoint to resume/evaluate")
    parser.add_argument("--model", default="vit_base_patch16", help="External ImageMAE model factory")
    parser.add_argument("--input-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--drop-path-rate", type=float, default=0.1)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--class-column", default="class")
    parser.add_argument("--path-column", default="file")
    parser.add_argument("--privacy-column", default="PRIVACY_FLAG")
    parser.add_argument("--split-column", default="split")
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--val-split", default="val")
    parser.add_argument("--test-split", default="test")
    parser.add_argument("--freeze-backbone", action="store_true", help="Train only the classification head")
    parser.add_argument("--cls-token", action="store_true", help="Use CLS token instead of global pooling")
    parser.add_argument("--eval", action="store_true", help="Evaluate --resume on the test split")
    return parser


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _run_epoch(model, loader, device, criterion, optimizer=None) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    correct = 0
    total = 0
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            if training:
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * labels.size(0)
            correct += (logits.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)
    if total == 0:
        raise ValueError("Cannot evaluate an empty dataloader")
    return {"loss": total_loss / total, "accuracy": correct / total}


def _save_checkpoint(path: Path, model, optimizer, epoch: int, class_to_idx: dict[str, int], args) -> None:
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "class_to_idx": class_to_idx,
            "args": vars(args),
        },
        path,
    )


def main(args: argparse.Namespace) -> None:
    if args.eval and not args.resume:
        raise ValueError("--eval requires --resume")
    _set_seed(args.seed)

    rows = read_manifest(args.metadata)
    class_to_idx = build_class_mapping(rows, args.class_column)
    train_rows = split_manifest(rows, split_column=args.split_column, split_name=args.train_split)
    val_rows = split_manifest(rows, split_column=args.split_column, split_name=args.val_split)
    test_rows = split_manifest(rows, split_column=args.split_column, split_name=args.test_split)

    def make_dataset(rows, is_train):
        return ImageMAEClassificationDataset(
            rows,
            image_root=args.image_root,
            generated_root=args.generated_root,
            class_to_idx=class_to_idx,
            transform=build_transform(is_train, args.input_size),
            path_column=args.path_column,
            label_column=args.class_column,
            privacy_column=args.privacy_column,
        )

    train_dataset = make_dataset(train_rows, True)
    val_dataset = make_dataset(val_rows, False)
    test_dataset = make_dataset(test_rows, False)
    device = torch.device(args.device)

    model, _, interpolate_pos_embed = build_classifier(
        args.image_mae_root,
        args.model,
        len(class_to_idx),
        global_pool=not args.cls_token,
        drop_path_rate=args.drop_path_rate,
    )
    if args.finetune:
        missing, skipped = load_checkpoint(model, args.finetune, interpolate_pos_embed=interpolate_pos_embed)
        print(f"Loaded --finetune; missing={missing}, skipped={skipped}")
    model.to(device)

    if args.freeze_backbone:
        for name, parameter in model.named_parameters():
            parameter.requires_grad = name.startswith("head.")
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    start_epoch = 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model"])
        if not args.eval and "optimizer" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])
            start_epoch = int(checkpoint.get("epoch", -1)) + 1

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "class_to_idx.json").write_text(json.dumps(class_to_idx, indent=2, sort_keys=True) + "\n")
    criterion = torch.nn.CrossEntropyLoss()
    if args.eval:
        stats = _run_epoch(
            model,
            DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers),
            device,
            criterion,
        )
        print(f"test loss={stats['loss']:.4f} accuracy={stats['accuracy']:.4f}")
        return

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    best_accuracy = float("-inf")
    for epoch in range(start_epoch, args.epochs):
        train_stats = _run_epoch(model, train_loader, device, criterion, optimizer)
        val_stats = _run_epoch(model, val_loader, device, criterion)
        print(
            f"epoch={epoch + 1} train_loss={train_stats['loss']:.4f} "
            f"train_accuracy={train_stats['accuracy']:.4f} "
            f"val_loss={val_stats['loss']:.4f} val_accuracy={val_stats['accuracy']:.4f}"
        )
        _save_checkpoint(output_dir / "latest.pt", model, optimizer, epoch, class_to_idx, args)
        if val_stats["accuracy"] > best_accuracy:
            best_accuracy = val_stats["accuracy"]
            _save_checkpoint(output_dir / "best.pt", model, optimizer, epoch, class_to_idx, args)


if __name__ == "__main__":
    main(get_args_parser().parse_args())
