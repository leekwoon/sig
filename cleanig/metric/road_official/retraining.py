"""Evaluation helpers adapted from the official ROAD implementation."""

from __future__ import annotations

import torch

from .utils import use_device


def road_eval(model, testloader, device: str | None = None):
    device = device or use_device
    correct = 0
    prob = 0.0

    model.eval()
    model.to(device)
    with torch.no_grad():
        for inputs, labels, predictions in testloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            predictions = torch.as_tensor(predictions, device=device, dtype=torch.long)

            outputs = model(inputs)
            predicted = outputs.argmax(dim=1)
            correct += (predicted == labels).sum().item()

            probs = torch.softmax(outputs, dim=1)
            prob += probs.gather(1, predictions.view(-1, 1)).sum().item()

    acc_avg = correct / len(testloader.dataset)
    prob_avg = prob / len(testloader.dataset)
    return acc_avg, prob_avg
