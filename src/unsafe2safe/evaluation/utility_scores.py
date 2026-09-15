"""Utility metrics used for downstream classification evaluation."""

import torch


def top1_accuracy(model, dataloader, device="cpu") -> float:
    """Return top-1 accuracy for a model over a labeled dataloader."""
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            outputs = model(inputs.to(device))
            if isinstance(outputs, (tuple, list)):
                outputs = outputs[-1]
            predictions = outputs.argmax(dim=1)
            correct += (predictions == labels.to(device)).sum().item()
            total += labels.size(0)
    return correct / total if total else 0.0
