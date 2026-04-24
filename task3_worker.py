"""Worker processes for parallel Task3 hyperparameter tuning experiments.

Each worker runs in its own process (via ``spawn`` context) with independent
CUDA context and data loaders, enabling concurrent training of up to 9 LeNet
experiments on a single GPU.
"""

from contextlib import redirect_stdout
from pathlib import Path
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import torchvision
from torch.utils.data import DataLoader, Subset

from my_utils import (
    train_model,
    create_learning_rate_scheduler,
    get_cifar10_data_augmentation,
)


class LeNet(nn.Module):
    """LeNet-style CNN for CIFAR-10 image classification.

    Consists of two convolutional layers and three fully connected layers.
    Adapted for 32x32 RGB input images.
    """

    def __init__(self, dropout: Optional[float] = None) -> None:
        super().__init__()

        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        self.dropout: Optional[nn.Dropout] = None
        if dropout is not None:
            if isinstance(dropout, float) and 0 < dropout < 1:
                self.dropout = nn.Dropout(dropout)
            else:
                raise ValueError("dropout must be a float between 0 and 1")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network.

        Args:
            x: Input tensor of shape (batch_size, 3, 32, 32).

        Returns:
            Output tensor of shape (batch_size, 10).
        """
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(x.size()[0], -1)
        x = F.relu(self.fc1(x))
        if self.dropout is not None:
            x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def train_experiment(
    exp_idx: int,
    lr: float,
    wd: float,
    ls: float,
    bs: int,
    train_indices: List[int],
    val_indices: List[int],
    num_epochs: int,
    ckpt_dir: str,
    log_dir: str,
) -> dict:
    """Train a single LeNet with given hyperparameters in a subprocess.

    Each call creates a fresh model, optimizer, data loader, and trains
    independently.  Training progress is written to a log file inside
    ``log_dir``.

    Args:
        exp_idx: Experiment index (1-based) for checkpoint/log naming.
        lr: Learning rate.
        wd: Weight decay (L2 regularization).
        ls: Label smoothing epsilon.
        bs: Batch size.
        train_indices: Indices into the full CIFAR-10 training set for
            training samples.
        val_indices: Indices into the full CIFAR-10 training set for
            validation samples.
        num_epochs: Maximum number of training epochs.
        ckpt_dir: Directory for model checkpoints.
        log_dir: Directory for training log files.

    Returns:
        Dict with keys: ``experiment_id``, ``learning_rate``, ``weight_decay``,
        ``label_smoothing``, ``batch_size``, ``best_val_accuracy``,
        ``final_train_loss``, ``epochs_trained``, ``optimizer``, ``log_file``.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Data transforms (light augmentation — same as notebook setup)
    transform_train, transform_test = get_cifar10_data_augmentation(style="light")

    # Load CIFAR-10 (already downloaded by the main notebook process)
    trainset = torchvision.datasets.CIFAR10(
        root="./dataset", train=True, download=False, transform=transform_train
    )
    valset = torchvision.datasets.CIFAR10(
        root="./dataset", train=True, download=False, transform=transform_test
    )

    train_loader = DataLoader(
        Subset(trainset, train_indices),
        batch_size=bs,
        shuffle=True,
        num_workers=1,
    )
    val_loader = DataLoader(
        Subset(valset, val_indices),
        batch_size=bs,
        shuffle=False,
        num_workers=1,
    )

    # Fresh model (no dropout — baseline LeNet for Task3)
    model = LeNet().to(device)

    # Adam optimizer with L2 regularization via weight_decay
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    criterion = nn.CrossEntropyLoss(label_smoothing=ls)

    # Cosine annealing scheduler with linear warm-up
    scheduler_config = create_learning_rate_scheduler(
        optimizer,
        scheduler_type="cosine",
        total_epochs=num_epochs,
        initial_lr=lr,
    )

    log_file = Path(log_dir) / f"exp_{exp_idx:02d}_lr{lr}_wd{wd}_ls{ls}_bs{bs}.log"

    # Train with stdout redirected to the log file
    with open(log_file, "w") as f:
        with redirect_stdout(f):
            train_losses, val_accuracies = train_model(
                model=model,
                train_loader=train_loader,
                criterion=criterion,
                optimizer=optimizer,
                num_epochs=num_epochs,
                save_path=str(Path(ckpt_dir) / f"exp_{exp_idx:02d}"),
                val_loader=val_loader,
                early_stopping_patience=10,
                gradient_clip=1.0,
                scheduler_config=scheduler_config,
            )

    best_val_acc = max(val_accuracies) if val_accuracies else 0.0
    final_loss = train_losses[-1] if train_losses else float("inf")
    epochs_done = len(train_losses)

    return {
        "experiment_id": exp_idx,
        "learning_rate": lr,
        "weight_decay": wd,
        "label_smoothing": ls,
        "batch_size": bs,
        "best_val_accuracy": best_val_acc,
        "final_train_loss": final_loss,
        "epochs_trained": epochs_done,
        "optimizer": "Adam",
        "log_file": str(log_file),
    }
