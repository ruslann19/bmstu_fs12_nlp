import collections
import torch
from torch import nn
from typing import Dict, Callable


class WeightedMSELoss(nn.Module):
    def __init__(
        self,
        classes_counts: Dict[int, int],
    ) -> None:
        super().__init__()
        classes_counts = collections.OrderedDict(sorted(classes_counts.items()))

        self.classes_counts = classes_counts
        total = sum(classes_counts.values())

        self.weights = {}
        for class_id, count in classes_counts.items():
            self.weights[class_id] = total / count

        # print("Веса:", self.weights)

    def forward(
        self,
        predictions: torch.tensor,
        targets: torch.tensor,
    ) -> torch.tensor:
        squared_errors = (predictions - targets) ** 2

        batch_weights = torch.ones_like(targets)
        for cls, weight in self.weights.items():
            mask = targets == cls
            batch_weights[mask] = weight

        return (squared_errors * batch_weights).mean()


def create_loss_fn(
    loss_type: str,
    classes_counts: Dict[int, int],
) -> Callable:
    match loss_type:
        case "MSELoss":
            return nn.MSELoss()
        case "WeightedMSELoss":
            return WeightedMSELoss(classes_counts)
