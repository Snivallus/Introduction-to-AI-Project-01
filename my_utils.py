"""Utility functions for CIFAR-10 image classification project.

This module contains common utilities for training, evaluation, and visualization
of neural networks on the CIFAR-10 dataset. Functions are type-annotated and
follow the project's coding style guidelines.

Example:
    >>> from my_utils import get_device, evaluate
    >>> device = get_device()
    >>> result = evaluate(model, test_loader)
    >>> print(f"Accuracy: {result['accuracy']:.2f}%")
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as transforms
from torchvision.transforms import RandomApply
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support
from contextlib import redirect_stdout
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Tuple
)
from my_model import LeNet
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts


CIFAR_10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_10_STD = (0.2470, 0.2435, 0.2616)
CIFAR_10_CLASS = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

FIGURE_DIR = Path('figures')
FIGURE_DIR.mkdir(parents=True, exist_ok=True)


def get_device(use_cuda: bool = True) -> torch.device:
    """Get the available device for PyTorch computations.

    Args:
        use_cuda: Whether to try using CUDA (GPU) if available.

    Returns:
        torch.device object representing the computation device.

    Example:
        >>> device = get_device()
        >>> print(f"Using device: {device}")
        Using device: cuda:0 (if GPU available)
    """
    if use_cuda and torch.cuda.is_available():
        device = torch.device("cuda:0")
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    return device


def get_cifar10_data_augmentation(
    style: str = 'light'
) -> Tuple[transforms.Compose, transforms.Compose]:
    """Get data augmentation transforms for CIFAR-10.

    Args:
        style: Augmentation style.
            ``'light'`` — basic flips and crops only (suitable for small models
            like LeNet).
            ``'full'`` — aggressive augmentation with ColorJitter, rotation,
            and random erasing (suitable for modern CNNs like ResNet).

    Returns:
        Tuple of (transform_train, transform_test) where:
        - transform_train: Transformations for training data (with augmentation)
        - transform_test: Transformations for test data (basic normalization)

    Raises:
        ValueError: If ``style`` is not ``'light'`` or ``'full'``.
    """
    # Test transform is the same regardless of style: no augmentation
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_10_MEAN, CIFAR_10_STD)
    ])

    if style == 'light':
        # Gentle augmentation: flips and crops only.
        # Heavier transforms (ColorJitter, Rotation, RandomErasing) are too
        # aggressive for small models like LeNet (~60K params).
        P = 0.1
        transform_train = transforms.Compose([
            transforms.RandomHorizontalFlip(p=P),
            # transforms.RandomVerticalFlip(p=P),  # Vertical flips are less common for natural images, so we omit them. 
            RandomApply([transforms.RandomCrop(32, padding=4, padding_mode='reflect')], p=P),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR_10_MEAN, CIFAR_10_STD)
        ])
    elif style == 'full':
        # Aggressive augmentation suitable for large-capacity models.
        P = 0.2
        transform_train = transforms.Compose([
            transforms.RandomHorizontalFlip(p=P),
            # transforms.RandomVerticalFlip(p=P),  # Vertical flips are less common for natural images, so we omit them. 
            RandomApply([transforms.RandomCrop(32, padding=4, padding_mode='reflect')], p=P),
            RandomApply([transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)], p=P),
            transforms.ToTensor(),
            RandomApply([transforms.GaussianBlur(kernel_size=(3, 3), sigma=(0.2, 0.5))], p=P),
            RandomApply([transforms.RandomRotation(5, fill=tuple(CIFAR_10_MEAN))], p=P),
            transforms.RandomErasing(p=P, scale=(0.02, 0.1), ratio=(0.3, 3.3), value=tuple(CIFAR_10_MEAN)),
            transforms.Normalize(CIFAR_10_MEAN, CIFAR_10_STD)
        ])
    else:
        raise ValueError(f"Unknown style: '{style}'. Use 'light' or 'full'.")

    return transform_train, transform_test


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    num_epochs: int,
    save_path: str,
    val_loader: Optional[DataLoader] = None,
    early_stopping_patience: Optional[int] = None,
    gradient_clip: Optional[float] = 1.0,
    print_every: Optional[int] = None,
    scheduler_config: Optional[Dict[str, Any]] = None
) -> Tuple[List[float], Optional[List[float]]]:
    """Train a neural network model with optional validation and early stopping.

    Args:
        model: PyTorch model to train.
        train_loader: DataLoader for training data.
        criterion: Loss function.
        optimizer: Optimizer for parameter updates.
        num_epochs: Number of training epochs.
        save_path: Directory to save model checkpoints.
        val_loader: DataLoader for validation data (optional).
        early_stopping_patience: Number of epochs to wait for improvement before early stopping (optional).
        gradient_clip: Maximum gradient norm for clipping (optional).
        print_every: Print training status every N batches (optional).
        scheduler_config: Dict returned by
            :func:`create_learning_rate_scheduler`. If provided, applies
            linear LR warm-up for the first ``warmup_epochs`` epochs, then
            steps the scheduler each epoch after warm-up.

    Returns:
        Tuple of (train_losses, val_accuracies) where:
        - train_losses: List of average training loss per epoch.
        - val_accuracies: List of validation accuracy per epoch (if val_loader provided).

    Raises:
        ValueError: If save_path is not a valid directory path.

    Example:
        >>> scheduler_cfg = create_learning_rate_scheduler(
        ...     optimizer, total_epochs=20, initial_lr=0.001)
        >>> train_losses, val_accuracies = train_model(
        ...     model, train_loader, criterion, optimizer, 20, 'checkpoints/',
        ...     val_loader=val_loader, scheduler_config=scheduler_cfg
        ... )
    """
    import os
    os.makedirs(save_path, exist_ok=True)

    train_losses: List[float] = []
    val_accuracies: Optional[List[float]] = [] if val_loader is not None else None

    best_val_accuracy = 0.0
    patience_counter = 0
    patience = early_stopping_patience if early_stopping_patience is not None else float('inf')
    restart_count = 0

    device = next(model.parameters()).device
    print(f"Training on device: {device}")

    # ── Initial evaluation (before any training) ────────────────────────
    # Verify that the initialized model produces reasonable baseline values.
    model.eval()
    with torch.no_grad():
        sample_batch = next(iter(train_loader))
        sample_inputs, sample_labels = sample_batch
        sample_inputs, sample_labels = sample_inputs.to(device), sample_labels.to(device)
        sample_outputs = model(sample_inputs)
        initial_loss = criterion(sample_outputs, sample_labels).item()

    print(f"Initial training loss (before training): {initial_loss:.4f}")

    if val_loader is not None:
        initial_val_accuracy = evaluate_accuracy(model, val_loader, device)
        print(f"Initial validation accuracy (before training): {initial_val_accuracy:.2f}%")

    print("-" * 50)
    # ───────────────────────────────────────────────────────────────────

    for epoch in range(num_epochs):
        # Learning rate warm-up and scheduling
        if scheduler_config is not None:
            warmup_epochs = scheduler_config['warmup_epochs']
            warmup_lr = scheduler_config['initial_lr']
            scheduler = scheduler_config['scheduler']

            if epoch < warmup_epochs:
                # Linear warm-up: scale LR from 0 to initial_lr
                lr = warmup_lr * (epoch + 1) / warmup_epochs
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr
            else:
                # Step the scheduler once per epoch after warm-up.
                # Detect restart via T_cur wrapping (CosineAnnealingWarmRestarts
                # increments T_cur each step and resets it to 0 at a restart).
                # This is robust to cycle_decay < 1 where the post-restart peak
                # LR may be lower than the pre-restart LR.
                old_T_cur = getattr(scheduler, 'T_cur', None)
                old_lr = optimizer.param_groups[0]['lr']

                scheduler.step()
                new_lr = optimizer.param_groups[0]['lr']

                if old_T_cur is not None and scheduler.T_cur < old_T_cur:
                    restart_count += 1
                    print(f"  >> Cosine annealing restart #{restart_count}: LR {old_lr:.6f} -> {new_lr:.6f}")

        running_loss = 0.0
        running_grad_norm = 0.0
        total_batches = 0

        # Training phase
        model.train()
        for i, data in enumerate(train_loader, 0):
            # 1. Get data
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            # 2. Zero gradients
            optimizer.zero_grad()

            # 3. Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # 4. Backward pass
            loss.backward()

            # 5. Gradient clipping (if specified) and norm tracking
            if gradient_clip is not None:
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=gradient_clip).item()
            else:
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float('inf')).item()
            running_grad_norm += grad_norm

            # 6. Optimizer step
            optimizer.step()

            # Record loss
            running_loss += loss.item()
            total_batches += 1

            # Print training status
            if print_every is not None and i % print_every == print_every - 1:
                avg_loss = running_loss / total_batches
                print(f'Epoch {epoch+1}: batch {i+1:5d} loss: {avg_loss:.3f}')

        # Calculate average training loss for the epoch
        epoch_avg_loss = running_loss / total_batches if total_batches > 0 else 0
        train_losses.append(epoch_avg_loss)

        # Compute average gradient norm and get current learning rate
        avg_grad_norm = running_grad_norm / total_batches if total_batches > 0 else 0.0
        current_lr = optimizer.param_groups[0]['lr']

        # Validation phase (if validation loader provided)
        if val_loader is not None:
            val_accuracy = evaluate_accuracy(model, val_loader, device)
            val_accuracies.append(val_accuracy)  # type: ignore

            print(f'Epoch {epoch+1}/{num_epochs}: '
                  f'Train Loss: {epoch_avg_loss:.4f}, '
                  f'Val Accuracy: {val_accuracy:.2f}%, '
                  f'Learning Rate: {current_lr:.6f}, '
                  f'Grad Norm: {avg_grad_norm:.4f}')

            # Early stopping logic
            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                patience_counter = 0
                # Save best model
                torch.save(model.state_dict(), f"{save_path}/best_model.pth")
                print(f'  -> New best model saved with accuracy: {val_accuracy:.2f}%')
            else:
                patience_counter += 1
                print(f'  -> No improvement for {patience_counter} epoch(s)')

            if patience_counter >= patience:
                print(f'Early stopping triggered at epoch {epoch+1}')
                break
        else:
            print(f'Epoch {epoch+1}/{num_epochs}: '
                  f'Train Loss: {epoch_avg_loss:.4f}, '
                  f'Learning Rate: {current_lr:.6f}, '
                  f'Grad Norm: {avg_grad_norm:.4f}')

    print('Finished Training')

    # Identify best epoch based on validation accuracy (if available) and print summary
    if val_accuracies:
        best_idx = val_accuracies.index(max(val_accuracies))
        best_epoch = best_idx + 1
        best_train_loss = train_losses[best_idx]
        print(
            f"Best validation accuracy: {val_accuracies[best_idx]:.2f}% "
            f"at epoch {best_epoch} with train loss: {best_train_loss:.4f}"
        )

    return train_losses, val_accuracies


def evaluate_accuracy(
    model: nn.Module,
    data_loader: DataLoader,
    device: Optional[torch.device] = None
) -> float:
    """Evaluate model accuracy on a dataset.

    Args:
        model: PyTorch model to evaluate.
        data_loader: DataLoader for the evaluation dataset.
        device: Device to evaluate on (if None, uses model's device).

    Returns:
        Accuracy percentage (0-100).

    Example:
        >>> accuracy = evaluate_accuracy(model, test_loader, device)
        >>> print(f"Test accuracy: {accuracy:.2f}%")
    """
    if device is None:
        device = next(model.parameters()).device

    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for data in data_loader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    return accuracy


def plot_loss_curves(
    train_losses: List[float],
    val_accuracies: Optional[List[float]] = None,
    save_path: str = 'loss_curve.png'
) -> None:
    """Plot training loss and validation accuracy curves.

    Args:
        train_losses: List of training losses per epoch.
        val_accuracies: List of validation accuracies per epoch (optional).
        save_path: Path to save the plot image.

    Example:
        >>> plot_loss_curves(train_losses, val_accuracies, 'task1_loss_curve.png')
    """
    fig, axes = plt.subplots(1, 2 if val_accuracies else 1, figsize=(10, 4))

    if val_accuracies:
        ax1, ax2 = axes
    else:
        ax1 = axes

    # Plot training loss curve
    ax1.plot(train_losses, 'b-', label='Training Loss', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss Curve')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot validation accuracy curve (if available)
    if val_accuracies:
        ax2.plot(val_accuracies, 'r-', label='Validation Accuracy', linewidth=2)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.set_title('Validation Accuracy Curve')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()

    print(f"Loss curve saved to {save_path}")
    print(f"Final training loss: {train_losses[-1]:.4f}")
    if val_accuracies:
        print(f"Final validation accuracy: {val_accuracies[-1]:.2f}%")


def count_parameters(model: nn.Module, decimals: int = 3) -> Dict[str, int]:
    """Count total and trainable parameters in a model.

    Args:
        model: PyTorch model.

    Returns:
        Dictionary with 'total' and 'trainable' parameter counts.

    Example:
        >>> counts = count_parameters(model)
        >>> print(f"Total parameters: {counts['total']:,}")
        >>> print(f"Trainable parameters: {counts['trainable']:,}")
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable_params = total_params - trainable_params

    return {
        'total': format_parameters(count=total_params, decimals=decimals),
        'trainable': format_parameters(count=trainable_params, decimals=decimals),
        'non_trainable': format_parameters(count=non_trainable_params, decimals=decimals),
    }


def format_parameters(count: int, decimals: int) -> str:
    """Convert a parameter count to a human-readable string with K/M/B suffix.

    Args:
        count: Number of parameters (integer).
        decimals: Number of decimal places to keep (default: 3).

    Returns:
        Formatted string, e.g. "6.432M", "123.4K", "1.2B", or "456" for small numbers.

    Example:
        >>> format_parameters(6432000)
        '6.432M'
        >>> format_parameters(123456)
        '123.456K'
    """
    for threshold, suffix in [(1e9, 'B'), (1e6, 'M'), (1e3, 'K')]:
        if count >= threshold:
            value = count / threshold
            return f"{value:.{decimals}f}".rstrip('0').rstrip('.') + suffix
    return str(count)


class CosineAnnealingWarmRestartsDecay(CosineAnnealingWarmRestarts):
    """Cosine annealing scheduler with decaying restarts.

    Extends :class:`torch.optim.lr_scheduler.CosineAnnealingWarmRestarts`
    by multiplying the base learning rates by ``cycle_decay`` each time
    a restart occurs.  This ensures each cycle starts with a lower peak
    LR, helping stabilise late-stage training in deep models.

    Args:
        optimizer: Wrapped optimizer.
        T_0: Number of epochs for the first restart.
        T_mult: Multiplication factor for T_0 after each restart.
        eta_min: Minimum learning rate.
        last_epoch: The index of the last epoch.
        cycle_decay: Factor multiplied into ``base_lrs`` at each restart.
            Clamped to ``[0.01, 1.0]``.  Default ``1.0`` (no decay).
    """
    def __init__(
        self,
        optimizer: optim.Optimizer,
        T_0: int,
        T_mult: int = 2,
        eta_min: float = 0.0,
        last_epoch: int = -1,
        cycle_decay: float = 1.0,
    ) -> None:
        if cycle_decay < 0.01 or cycle_decay > 1.0:
            import warnings
            warnings.warn(
                f"cycle_decay={cycle_decay} is outside [0.01, 1.0]; "
                f"clamping to {max(0.01, min(1.0, cycle_decay))}"
            )
        self.cycle_decay = max(0.01, min(1.0, cycle_decay))
        self._restart_count = 0
        super().__init__(optimizer, T_0, T_mult, eta_min, last_epoch)

    def step(self, epoch: Optional[int] = None) -> None:
        # First call: delegate without decay logic
        if self.last_epoch == -1:
            super().step(epoch)
            return

        # Snapshot before stepping
        old_T_cur = self.T_cur

        # Delegate to parent
        super().step(epoch)

        # Detect restart: T_cur was reset to 0 after reaching T_i.
        # super().step() above already set the optimizer LR using the
        # undecayed base_lrs.  We must re-apply with the decayed values
        # so the current epoch immediately reflects the lower peak.
        if self.T_cur < old_T_cur:
            self.base_lrs = [lr * self.cycle_decay for lr in self.base_lrs]
            # At T_cur=0, get_lr() returns base_lrs (cos(0)=1), so
            # assign directly to avoid the "use get_last_lr()" warning.
            for param_group, lr in zip(self.optimizer.param_groups, self.base_lrs):
                param_group['lr'] = lr
            self._last_lr = [group['lr'] for group in self.optimizer.param_groups]
            self._restart_count += 1

    def get_restart_count(self) -> int:
        """Return the number of restarts that have occurred."""
        return self._restart_count


def create_learning_rate_scheduler(
    optimizer: optim.Optimizer,
    scheduler_type: str = 'cosine',
    total_epochs: int = 20,
    warmup_epochs: Optional[int] = None,
    initial_lr: float = 0.001,
    min_lr: float = 1e-6,
    T_0: Optional[int] = None,
    T_mult: int = 2,
    gamma: float = 0.1,
    cycle_decay: float = 1.0,
) -> Dict[str, Any]:
    """Create a learning rate scheduler with linear warm-up configuration.

    Returns a dictionary with scheduler and warm-up parameters, designed
    to be passed to ``train_model()`` via its ``scheduler_config`` parameter.
    The caller should not call ``scheduler.step()`` manually — ``train_model``
    handles both warm-up and scheduler stepping automatically.

    Args:
        optimizer: Optimizer to schedule.
        scheduler_type: Type of scheduler ('cosine', 'step').
        total_epochs: Total number of training epochs.
        warmup_epochs: Number of warm-up epochs. If None, defaults to min(5, max(2, 10% of total_epochs)).
        initial_lr: Initial learning rate (used to scale warm-up).
        min_lr: Minimum learning rate (eta_min).
        T_0: Period for first restart in epochs. If None, defaults to total_epochs // 2.
        T_mult: Multiplication factor for T_0 after each restart.
        gamma: Multiplicative factor for step LR scheduler (if scheduler_type='step').
        cycle_decay: Factor applied to ``base_lrs`` at each cosine restart.
            Must be in ``[0.01, 1.0]``.  Default ``1.0`` (no decay).  Ignored for non-cosine schedulers.

    Returns:
        Dictionary with keys:
        - ``scheduler``: PyTorch learning rate scheduler.
        - ``warmup_epochs``: Number of warm-up epochs.
        - ``initial_lr``: Initial learning rate.

    Example:
        >>> config = create_learning_rate_scheduler(
        ...     optimizer, scheduler_type='cosine',
        ...     total_epochs=20, initial_lr=0.001
        ... )
        >>> train_losses, val_accs = train_model(
        ...     model, loader, criterion, optimizer, 20, 'ckpt/',
        ...     scheduler_config=config
        ... )
    """
    if warmup_epochs is None:
        warmup_epochs = min(5, max(2, int(total_epochs * 0.1)))
    if T_0 is None:
        T_0 = max(1, total_epochs // 2)

    # Suppress the "was invoked with deprecated argument" spam from PyTorch
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        if scheduler_type == 'cosine':
            scheduler = CosineAnnealingWarmRestartsDecay(
                optimizer, T_0=T_0, T_mult=T_mult,
                eta_min=min_lr, cycle_decay=cycle_decay
            )
            scheduler_desc = (
                f"CosineAnnealingWarmRestartsDecay (T_0={T_0}, T_mult={T_mult}, "
                f"cycle_decay={cycle_decay})"
            )
        elif scheduler_type == 'step':
            from torch.optim.lr_scheduler import StepLR
            step_size = max(1, total_epochs // 3)
            scheduler = StepLR(optimizer, step_size=step_size, gamma={gamma})
            scheduler_desc = f"StepLR (step_size={step_size}, gamma={gamma})"
        else:
            raise ValueError(f"Unknown scheduler type: {scheduler_type}")

    print(f"Created {scheduler_desc}")
    print(f"  Warm-up: {warmup_epochs} epoch(s), Initial LR: {initial_lr}")
    print(f"  Min LR: {min_lr}, Total epochs: {total_epochs}")

    return {
        'scheduler': scheduler,
        'warmup_epochs': warmup_epochs,
        'initial_lr': initial_lr,
    }


# ═══════════════════════════════════════════════════════════════════════
# Task3: Hyperparameter Tuning Helpers
# ═══════════════════════════════════════════════════════════════════════


def train_experiment(
    exp_idx: int,
    dropout: Optional[float],
    lr: float,
    wd: float,
    ls: float,
    bs: int,
    train_indices: List[int],
    val_indices: List[int],
    num_epochs: int,
    min_lr: float,
    T_0: int,
    T_mult: int,
    cycle_decay: float,
    momentum: float,
    ckpt_dir: str,
    log_dir: str,
) -> Dict[str, Any]:
    """Train a single LeNet with dropout for Task3 hyperparameter tuning.

    Each call creates data loaders, optimizer, and trains independently.  
    Training progress is written to a log file inside ``log_dir``.  
    After training, evaluates on the CIFAR-10 test set via
    :func:`evaluate` and records the test accuracy.

    Args:
        exp_idx: Experiment index (1-based) for checkpoint/log naming.
        lr: Learning rate.
        wd: Weight decay (L2 regularization).
        ls: Label smoothing epsilon.
        bs: Batch size.
        train_indices: Indices into the full CIFAR-10 training set for training samples.
        val_indices: Indices into the full CIFAR-10 training set for validation samples.
        num_epochs: Maximum number of training epochs.
        T_0: Number of epochs for the first cosine annealing restart.
        T_mult: Multiplication factor for T_0 after each restart.
        cycle_decay: Factor to decay the learning rate at each cosine restart (0.01 to 1.0).
        momentum: Momentum factor for SGD optimizer.
        ckpt_dir: Directory for model checkpoints.
        log_dir: Directory for training log files.

    Returns:
        Dict with keys: ``experiment_id``, ``learning_rate``, ``weight_decay``,
        ``label_smoothing``, ``batch_size``, ,
        ``best_val_accuracy``, ``test_accuracy``, ``final_train_loss``,
        ``epochs_trained``, ``optimizer``, ``log_file``.
    """
    log_file = Path(log_dir) / f"exp_{exp_idx:02d}_lr{lr}_wd{wd}_ls{ls}_bs{bs}.log"

    # Redirect stdout to the log file
    with open(log_file, "w") as f:
        with redirect_stdout(f):

            # Move model to device
            device = get_device()
            model = LeNet(dropout=dropout).to(device)

            # Data transforms (light augmentation — same as notebook setup)
            transform_train, transform_test = get_cifar10_data_augmentation(style="light")

            # Load CIFAR-10 (already downloaded by the main notebook process)
            trainset = torchvision.datasets.CIFAR10(
                root="./dataset", train=True, download=False, transform=transform_train
            )
            valset = torchvision.datasets.CIFAR10(
                root="./dataset", train=True, download=False, transform=transform_test
            )
            testset = torchvision.datasets.CIFAR10(
                root="./dataset", train=False, download=False, transform=transform_test
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
            test_loader = DataLoader(
                testset, batch_size=bs, shuffle=False, num_workers=1,
            )

            # SGD optimizer with L2 regularization via weight_decay
            optimizer = optim.SGD(
                model.parameters(), 
                lr=lr, 
                weight_decay=wd,
                momentum=momentum,
                nesterov=True
            )
            criterion = nn.CrossEntropyLoss(label_smoothing=ls)

            # Cosine annealing scheduler with linear warm-up
            scheduler_config = create_learning_rate_scheduler(
                optimizer,
                scheduler_type="cosine",
                total_epochs=num_epochs,
                initial_lr=lr,
                min_lr=min_lr,
                T_0=T_0,
                T_mult=T_mult,
                cycle_decay=cycle_decay
            )

            # Train model and record training losses and validation accuracies
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

            # Evaluate on test set using my_utils.evaluate
            test_result = evaluate(
                model, test_loader, device,
                model_name=f"exp_{exp_idx:02d}",
                verbose=False,
                skip_mc_dropout=True,
            )
            test_accuracy = test_result['accuracy']

    return {
        "experiment_id": exp_idx,
        "learning_rate": lr,
        "weight_decay": wd,
        "label_smoothing": ls,
        "batch_size": bs,
        "best_val_accuracy": best_val_acc,
        "test_accuracy": test_accuracy,
        "final_train_loss": final_loss,
        "epochs_trained": epochs_done,
        "optimizer": "Adam",
        "log_file": str(log_file),
    }


def plot_task3_hyperparameter_effects(
    results_df: "pd.DataFrame",
    save_dir: str = 'figures',
    figure_prefix: str = 'task3',
    metric: str = 'test_accuracy',
) -> None:
    """Plot line charts showing each hyperparameter's systematic effect on accuracy.

    For each hyperparameter (learning_rate, weight_decay, label_smoothing,
    batch_size), creates a figure with:
    - A line connecting the mean accuracy across all combinations at each
      parameter value
    - Error bars showing ±1 standard deviation
    - Individual experiment results overlaid as scatter points

    Args:
        results_df: DataFrame containing experiment results. Must include
            columns ``learning_rate``, ``weight_decay``, ``label_smoothing``,
            ``batch_size``, and the column specified by ``metric``.
        save_dir: Directory to save the output PDF plots.
        figure_prefix: Prefix for output filenames.
        metric: Column name in ``results_df`` holding the accuracy metric
            to plot (default ``'test_accuracy'``).
    """
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    param_configs = [
        ('learning_rate', 'Learning Rate'),
        ('weight_decay', 'Weight Decay'),
        ('label_smoothing', 'Label Smoothing'),
        ('batch_size', 'Batch Size'),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    for ax, (param_col, param_label) in zip(axes, param_configs):
        if param_col not in results_df.columns:
            ax.text(0.5, 0.5, f"No data for\n{param_col}",
                    ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'Effect of {param_label}')
            continue

        # Group by parameter value and compute statistics
        grouped = results_df.groupby(param_col)[metric]
        means = grouped.mean()
        stds = grouped.std()

        # X-axis positions
        x = range(len(means))

        # Line plot with error bars
        ax.errorbar(
            x, means.values, yerr=stds.values,
            fmt='-o', capsize=5, capthick=2, linewidth=2,
            markersize=8, color='#2196F3', ecolor='#FF5722',
            label=f'Mean ± Std',
        )

        # Overlay individual points with small jitter
        for xi, (val, group) in enumerate(grouped):
            jitter = np.random.default_rng(42).uniform(-0.05, 0.05, len(group))
            ax.scatter(
                xi + jitter, group.values,
                alpha=0.5, color='#FF5722', zorder=5, s=30,
                label='Individual' if xi == 0 else '',
            )

        # Format tick labels
        tick_labels = []
        for v in means.index:
            if isinstance(v, float) and v < 0.001:
                tick_labels.append(f'{v:.1e}')
            elif isinstance(v, float):
                tick_labels.append(f'{v:.4f}')
            else:
                tick_labels.append(str(v))

        ax.set_xticks(x)
        ax.set_xticklabels(tick_labels)
        ax.set_xlabel(param_label)
        ax.set_ylabel('Accuracy (%)')
        ax.set_title(f'Effect of {param_label} on {metric}')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.suptitle('Task3: Hyperparameter Effect Analysis', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path / f'{figure_prefix}_hyperparameter_effects.pdf', dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Saved: {save_path / f'{figure_prefix}_hyperparameter_effects.pdf'}")

    # ── Second figure: LR × BS interaction grouped bar chart ──────────────
    if 'learning_rate' in results_df.columns and 'batch_size' in results_df.columns:
        fig2, ax2 = plt.subplots(figsize=(9, 6))

        # Aggregate over the remaining hyperparameters (WD, LS)
        grouped = results_df.groupby(['learning_rate', 'batch_size'])[metric]
        means = grouped.mean().unstack(level='batch_size')
        stds = grouped.std().unstack(level='batch_size')

        n_lr = means.shape[0]
        n_bs = means.shape[1]
        bar_width = 0.8 / n_lr
        x = np.arange(n_bs)

        colors = ['#2196F3', '#FF9800', '#4CAF50']
        for i, lr_val in enumerate(means.index):
            offset = (i - (n_lr - 1) / 2) * bar_width
            ax2.bar(
                x + offset, means.loc[lr_val].values,
                yerr=stds.loc[lr_val].values,
                width=bar_width, label=f'lr = {lr_val}',
                color=colors[i % len(colors)], capsize=4,
                edgecolor='white', linewidth=0.8,
                error_kw={'linewidth': 1.5},
            )

        ax2.set_xticks(x)
        tick_labels_bs = []
        for v in means.columns:
            if isinstance(v, float) and v < 0.001:
                tick_labels_bs.append(f'{v:.1e}')
            else:
                tick_labels_bs.append(str(v))
        ax2.set_xticklabels(tick_labels_bs)
        ax2.set_xlabel('Batch Size')
        ax2.set_ylabel('Accuracy (%)')
        ax2.set_title(f'LR × BS Interaction Effect on {metric}', fontsize=13, fontweight='bold')
        ax2.legend(title='Learning Rate', fontsize=9)
        ax2.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        plt.savefig(save_path / f'{figure_prefix}_lr_bs_interaction.pdf', dpi=300, bbox_inches='tight')
        plt.show()
        print(f"Saved: {save_path / f'{figure_prefix}_lr_bs_interaction.pdf'}")


def _has_dropout(model: nn.Module) -> bool:
    """Check whether a model contains at least one dropout layer.

    Args:
        model: PyTorch model.

    Returns:
        True if any module is an instance of ``nn.Dropout``.
    """
    return any(isinstance(m, nn.Dropout) for m in model.modules())


def _collect_predictions(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run inference on a dataloader and collect all predictions.

    Args:
        model: PyTorch model (will be set to eval mode).
        data_loader: DataLoader yielding (inputs, labels).
        device: Computation device.

    Returns:
        Tuple ``(predictions, probabilities, labels)`` — each as a
        :class:`numpy.ndarray`.
    """
    model.eval()
    all_logits: List[torch.Tensor] = []
    all_labels: List[torch.Tensor] = []

    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs = inputs.to(device, non_blocking=True)
            outputs = model(inputs)
            all_logits.append(outputs.cpu())
            all_labels.append(labels)

    logits = torch.cat(all_logits, dim=0)
    labels = torch.cat(all_labels, dim=0).numpy()
    probs = torch.softmax(logits, dim=1).numpy()
    predictions = np.argmax(probs, axis=1)

    return predictions, probs, labels


# ------------------------------------------------------------------
# 1. Per-class metrics (confusion matrix, F1, precision, recall)
# ------------------------------------------------------------------

def compute_per_class_metrics(
    predictions: np.ndarray,
    targets: np.ndarray,
    class_names: Tuple[str, ...] = CIFAR_10_CLASS,
) -> Dict[str, Dict[str, float]]:
    """Compute per-class precision, recall, and F1-score.

    Falls back to a manual implementation when ``sklearn`` is not available.

    Args:
        predictions: Predicted class indices.
        targets: Ground truth class indices.
        class_names: Class name tuple (length = number of classes).

    Returns:
        Dict mapping class name to ``{'precision', 'recall', 'f1'}``.
    """
    num_classes = len(class_names)
    prec, rec, f1, _ = precision_recall_fscore_support(
        targets, predictions, labels=range(num_classes), zero_division=0
    )
    return {
        name: {
            'precision': float(p),
            'recall': float(r),
            'f1': float(f),
        }
        for name, p, r, f in zip(class_names, prec, rec, f1)
    }


def compute_confusion_matrix(
    predictions: np.ndarray,
    targets: np.ndarray,
    num_classes: int = 10,
) -> np.ndarray:
    """Compute a confusion matrix.

    Args:
        predictions: Predicted class indices.
        targets: Ground truth class indices.
        num_classes: Number of classes.

    Returns:
        ``(num_classes, num_classes)`` integer confusion matrix where
        entry ``[i, j]`` is the number of samples from class **i**
        predicted as class **j**.
    """
    try:
        from sklearn.metrics import confusion_matrix
        return confusion_matrix(targets, predictions, labels=range(num_classes))
    except ImportError:
        cm = np.zeros((num_classes, num_classes), dtype=int)
        for t, p in zip(targets, predictions):
            cm[t, p] += 1
        return cm


# ------------------------------------------------------------------
# 2. Expected Calibration Error (ECE) + Reliability Diagram data
# ------------------------------------------------------------------

def compute_ece(
    probabilities: np.ndarray,
    targets: np.ndarray,
    n_bins: int = 10,
) -> Tuple[float, List[float], List[float], List[int]]:
    """Compute Expected Calibration Error (ECE).

    Partitions predictions by confidence into ``n_bins`` bins, then
    computes the weighted average ``|accuracy - confidence|`` across
    bins.

    Args:
        probabilities: Softmax probabilities of shape ``(n, num_classes)``.
        targets: Ground truth class indices of shape ``(n,)``.
        n_bins: Number of equal-width confidence bins in ``[0, 1]``.

    Returns:
        Tuple ``(ece, bin_confidences, bin_accuracies, bin_counts)``.
    """
    confidences = np.max(probabilities, axis=1)
    predictions = np.argmax(probabilities, axis=1)
    accuracies = (predictions == targets).astype(float)

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    bin_confidences: List[float] = []
    bin_accuracies: List[float] = []
    bin_counts: List[int] = []

    for i in range(n_bins):
        in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
        count = int(in_bin.sum())
        bin_counts.append(count)
        if count > 0:
            bin_conf = float(confidences[in_bin].mean())
            bin_acc = float(accuracies[in_bin].mean())
        else:
            bin_conf = float((bin_boundaries[i] + bin_boundaries[i + 1]) / 2)
            bin_acc = 0.0
        bin_confidences.append(bin_conf)
        bin_accuracies.append(bin_acc)
        ece += abs(bin_acc - bin_conf) * count / len(confidences)

    return ece, bin_confidences, bin_accuracies, bin_counts


# ------------------------------------------------------------------
# 3. Prediction distribution KL divergence
# ------------------------------------------------------------------

def compute_prediction_distribution_kl(
    predictions: np.ndarray,
    targets: np.ndarray,
    num_classes: int = 10,
) -> float:
    """Compute KL divergence ``D_KL(P_true || P_pred)``.

    Measures how much the **predicted** class distribution diverges from
    the **true** test-set class distribution.  Additive (Laplace)
    smoothing avoids ``log(0)``.

    Args:
        predictions: Predicted class indices.
        targets: Ground truth class indices.
        num_classes: Number of classes.

    Returns:
        KL divergence in nats.  Values near 0 indicate the model's
        predictions follow the same class proportions as the test set.
    """
    pred_counts = np.bincount(predictions, minlength=num_classes).astype(float) + 1.0
    true_counts = np.bincount(targets, minlength=num_classes).astype(float) + 1.0
    pred_dist = pred_counts / pred_counts.sum()
    true_dist = true_counts / true_counts.sum()
    return float((true_dist * np.log(true_dist / pred_dist)).sum())


# ------------------------------------------------------------------
# 4. MC Dropout uncertainty estimation
# ------------------------------------------------------------------

def mc_dropout_evaluate(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    num_samples: int = 20,
) -> Dict[str, Any]:
    """Estimate predictive uncertainty via Monte Carlo Dropout.

    Keeps batch-norm layers in eval mode (using running statistics)
    while enabling dropout during all forward passes.  The variance
    across passes reflects the model's epistemic uncertainty.

    Args:
        model: PyTorch model containing at least one ``nn.Dropout`` layer.
        data_loader: DataLoader for evaluation.
        device: Computation device.
        num_samples: Number of stochastic forward passes.

    Returns:
        Dict with keys:

        - ``variance_mean`` — mean predictive variance across all samples
        - ``variance_std`` — standard deviation of predictive variances
        - ``variance_median`` — median predictive variance
        - ``accurate_var_mean`` — mean variance of *correctly* predicted samples
        - ``inaccurate_var_mean`` — mean variance of *incorrectly* predicted samples
        - ``accuracy`` — MC Dropout accuracy averaged over samples
        - ``sample_variances`` — per-sample variances (only when called from
          ``evaluate`` with ``save_plots=True``)

    Raises:
        ValueError: If the model has no dropout layers.
    """
    if not _has_dropout(model):
        raise ValueError(
            "Model has no dropout layers — MC Dropout requires at least one "
            "nn.Dropout module."
        )

    # --- Enable dropout; keep batch-norm in eval mode ---
    model.eval()
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.training = True

    # --- Collect multiple forward passes ---
    all_samples: List[torch.Tensor] = []
    with torch.no_grad():
        for _ in range(num_samples):
            batch_logits: List[torch.Tensor] = []
            for inputs, _ in data_loader:
                inputs = inputs.to(device, non_blocking=True)
                outputs = model(inputs)
                batch_logits.append(outputs.cpu())
            all_samples.append(torch.cat(batch_logits, dim=0))

    model.eval()  # restore full eval mode

    # --- Compute predictive mean and variance ---
    stacked = torch.stack(all_samples, dim=0)  # (num_samples, n, num_classes)
    probs = torch.softmax(stacked, dim=2).numpy()
    mean_probs = probs.mean(axis=0)
    var_probs = probs.var(axis=0)

    predicted = np.argmax(mean_probs, axis=1)
    sample_variances = np.array([
        var_probs[i, predicted[i]] for i in range(len(predicted))
    ])

    # --- Get labels for accuracy breakdown ---
    all_labels: List[torch.Tensor] = []
    model.eval()
    with torch.no_grad():
        for _, labels in data_loader:
            all_labels.append(labels)
    labels = torch.cat(all_labels).numpy()
    accurate = predicted == labels

    result: Dict[str, Any] = {
        'variance_mean': float(sample_variances.mean()),
        'variance_std': float(sample_variances.std()),
        'variance_median': float(np.median(sample_variances)),
        'accuracy': float(accurate.mean() * 100),
        'sample_variances': sample_variances,
        'accurate_mask': accurate,
    }

    if accurate.any():
        result['accurate_var_mean'] = float(sample_variances[accurate].mean())
    else:
        result['accurate_var_mean'] = 0.0

    if (~accurate).any():
        result['inaccurate_var_mean'] = float(sample_variances[~accurate].mean())
    else:
        result['inaccurate_var_mean'] = 0.0

    return result


# ------------------------------------------------------------------
# Plotting helpers
# ------------------------------------------------------------------

def _plot_confusion_matrix(
    cm: np.ndarray,
    class_names: Tuple[str, ...],
    model_name: str,
    plot_dir: Path,
) -> None:
    """Plot and save a normalized confusion matrix heatmap."""
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm_norm, cmap='Blues', vmin=0, vmax=1)

    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha='right')
    ax.set_yticklabels(class_names)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title(f'Normalized Confusion Matrix — {model_name}')

    for i in range(cm_norm.shape[0]):
        for j in range(cm_norm.shape[1]):
            val = cm_norm[i, j]
            color = 'white' if val > 0.5 else 'black'
            ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                    fontsize=7, color=color)

    fig.colorbar(im, ax=ax, fraction=0.046)
    plt.tight_layout()
    plt.savefig(plot_dir / f'{model_name}_confusion_matrix.pdf',
                dpi=150, bbox_inches='tight')
    plt.close()


def _plot_reliability_diagram(
    bin_confidences: List[float],
    bin_accuracies: List[float],
    bin_counts: List[int],
    ece: float,
    model_name: str,
    plot_dir: Path,
) -> None:
    """Plot and save a reliability diagram with gap bars."""
    fig, ax = plt.subplots(figsize=(7, 7))

    # Perfect calibration diagonal
    ax.plot([0, 1], [0, 1], '--k', alpha=0.5, label='Perfect Calibration')

    # Gap bars (red vertical lines showing |acc - conf|)
    bin_width = 1.0 / len(bin_confidences)
    for conf, acc in zip(bin_confidences, bin_accuracies):
        ax.plot([conf, conf], [acc, conf], 'r-', alpha=0.3, linewidth=1)

    # Bin bars
    ax.bar(bin_confidences, bin_accuracies, width=bin_width,
           alpha=0.6, color='#2196F3', edgecolor='white', linewidth=1,
           label='Model')

    ax.set_xlabel('Confidence')
    ax.set_ylabel('Accuracy')
    ax.set_title(f'Reliability Diagram — {model_name} (ECE = {ece:.4f})')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)

    # Annotate bin counts at the bottom
    for conf, count in zip(bin_confidences, bin_counts):
        if count > 0:
            ax.annotate(str(count), (conf, 0.02),
                        ha='center', fontsize=7, alpha=0.7)

    plt.tight_layout()
    plt.savefig(plot_dir / f'{model_name}_reliability_diagram.pdf',
                dpi=150, bbox_inches='tight')
    plt.close()


def _plot_mc_dropout_variance(
    dropout_result: Dict[str, Any],
    model_name: str,
    plot_dir: Path,
) -> None:
    """Plot MC Dropout variance histogram and accuracy breakdown."""
    variances = dropout_result['sample_variances']
    accurate = dropout_result['accurate_mask']

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # --- Left: full distribution ---
    axes[0].hist(variances, bins=50, alpha=0.7, color='purple', edgecolor='white')
    axes[0].axvline(dropout_result['variance_mean'], color='red',
                    linestyle='--', linewidth=2,
                    label=f"Mean: {dropout_result['variance_mean']:.6f}")
    axes[0].set_xlabel('Predictive Variance')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title(f'MC Dropout Variance — {model_name}')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # --- Right: correct vs. wrong ---
    if accurate.any() and (~accurate).any():
        axes[1].hist(variances[accurate], bins=30, alpha=0.6, color='green',
                     edgecolor='white',
                     label=f'Correct (mean={dropout_result["accurate_var_mean"]:.6f})')
        axes[1].hist(variances[~accurate], bins=30, alpha=0.6, color='red',
                     edgecolor='white',
                     label=f'Wrong (mean={dropout_result["inaccurate_var_mean"]:.6f})')
        axes[1].set_xlabel('Predictive Variance')
        axes[1].set_ylabel('Frequency')
        axes[1].set_title('Variance: Correct vs. Wrong Predictions')
        axes[1].legend(fontsize=9)
        axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(plot_dir / f'{model_name}_mc_dropout_variance.pdf',
                dpi=150, bbox_inches='tight')
    plt.close()


# ------------------------------------------------------------------
# Public API: evaluate
# ------------------------------------------------------------------

def evaluate(
    model: nn.Module,
    test_loader: DataLoader,
    device: Optional[torch.device] = None,
    class_names: Tuple[str, ...] = CIFAR_10_CLASS,
    model_name: str = 'model',
    save_plots: bool = False,
    plot_dir: str = 'figures',
    n_bins_ece: int = 10,
    mc_dropout_samples: int = 20,
    skip_mc_dropout: bool = False,
    verbose: bool = False,
) -> Dict[str, Any]:
    """Comprehensive evaluation of a CIFAR-10 classifier.

    Computes all five metric families discussed in the project report:

    1. **Confusion matrix + per-class F1** (via :func:`compute_confusion_matrix`
       and :func:`compute_per_class_metrics`).
    2. **ECE + reliability diagram** (via :func:`compute_ece`).
    3. **Prediction-distribution KL divergence** (via
       :func:`compute_prediction_distribution_kl`).
    4. **MC Dropout uncertainty** (via :func:`mc_dropout_evaluate`; only
       when the model contains dropout layers).

    Args:
        model: PyTorch model to evaluate.
        test_loader: DataLoader for the test set.
        device: Computation device.  If ``None``, inferred from model
            parameters.
        class_names: Tuple of class label names.
        model_name: Identifier used in printed output and plot filenames.
        save_plots: If ``True``, saves confusion matrix, reliability
            diagram, and (if applicable) MC Dropout variance plots to
            ``plot_dir``.
        plot_dir: Directory for saved plots.
        n_bins_ece: Number of confidence bins for ECE computation.
        mc_dropout_samples: Number of stochastic forward passes for MC
            Dropout (ignored if the model has no dropout or
            ``skip_mc_dropout=True``).
        skip_mc_dropout: If ``True``, skip MC Dropout evaluation even if
            the model has dropout layers.  Useful for bulk evaluation
            during hyperparameter tuning.
        verbose: If ``True``, print a results summary.

    Returns:
        Dict with the following keys:

        - ``model_name`` — identifier string
        - ``accuracy`` — top-1 accuracy (percent)
        - ``per_class_metrics`` — dict of class → ``{precision, recall, f1}``
        - ``confusion_matrix`` — ``(num_classes, num_classes)`` integer array
        - ``ece`` — Expected Calibration Error
        - ``reliability`` — dict with ``bin_confidences``, ``bin_accuracies``,
          ``bin_counts`` (for manual plotting)
        - ``pred_dist_kl`` — KL divergence in nats
        - ``true_class_distribution`` — dict of class name → count
        - ``pred_class_distribution`` — dict of class name → count
        - ``mc_dropout`` — dict from :func:`mc_dropout_evaluate`, or ``None``
          if the model has no dropout
        - ``predictions`` — ``(n,)`` numpy array of predicted class indices
        - ``ground_truth`` — ``(n,)`` numpy array of true class indices
        - ``probabilities`` — ``(n, num_classes)`` numpy softmax probabilities
    """
    if device is None:
        device = next(model.parameters()).device

    # ── 0. Header ────────────────────────────────────────────────────
    if verbose:
        print(f"\n{'=' * 70}")
        print(f"Comprehensive Evaluation — {model_name}")
        print(f"{'=' * 70}")

    # ── 1. Collect predictions ───────────────────────────────────────
    predictions, probabilities, targets = _collect_predictions(
        model, test_loader, device
    )

    # ── 2. Top-1 accuracy ────────────────────────────────────────────
    accuracy = float((predictions == targets).mean() * 100)

    # ── 3. Confusion matrix ──────────────────────────────────────────
    cm = compute_confusion_matrix(predictions, targets, len(class_names))

    # ── 4. Per-class metrics ─────────────────────────────────────────
    per_class = compute_per_class_metrics(predictions, targets, class_names)

    # ── 5. Expected Calibration Error ────────────────────────────────
    ece, bin_conf, bin_acc, bin_counts = compute_ece(
        probabilities, targets, n_bins=n_bins_ece
    )

    # ── 6. Prediction-distribution KL divergence ─────────────────────
    kl_div = compute_prediction_distribution_kl(
        predictions, targets, num_classes=len(class_names)
    )

    # ── 7. Class distributions ───────────────────────────────────────
    true_dist = {
        name: int((targets == i).sum()) for i, name in enumerate(class_names)
    }
    pred_dist = {
        name: int((predictions == i).sum()) for i, name in enumerate(class_names)
    }

    # ── 8. MC Dropout ────────────────────────────────────────────────
    dropout_result: Optional[Dict[str, Any]] = None
    if _has_dropout(model) and not skip_mc_dropout:
        if verbose:
            print("\n[MC Dropout] Evaluating uncertainty ...")
        dropout_result = mc_dropout_evaluate(
            model, test_loader, device, num_samples=mc_dropout_samples
        )
        # Strip large arrays from the result dict unless plots are requested
        if not save_plots:
            dropout_result.pop('sample_variances', None)
            dropout_result.pop('accurate_mask', None)

    # ── Assemble result dict ─────────────────────────────────────────
    result: Dict[str, Any] = {
        'model_name': model_name,
        'accuracy': accuracy,
        'per_class_metrics': per_class,
        'confusion_matrix': cm,
        'ece': ece,
        'reliability': {
            'bin_confidences': bin_conf,
            'bin_accuracies': bin_acc,
            'bin_counts': bin_counts,
        },
        'pred_dist_kl': kl_div,
        'true_class_distribution': true_dist,
        'pred_class_distribution': pred_dist,
        'mc_dropout': dropout_result,
        'predictions': predictions,
        'ground_truth': targets,
        'probabilities': probabilities,
    }

    # ── Print summary ────────────────────────────────────────────────
    if verbose:
        f1_scores = [v['f1'] for v in per_class.values()]
        avg_f1 = float(np.mean(f1_scores))

        print(f"\n{'─' * 70}")
        print(f"  Accuracy:                       {accuracy:.2f}%")
        print(f"  Average F1:                     {avg_f1:.4f}")
        print(f"  ECE (calibration error):        {ece:.4f}")
        print(f"  Pred distribution KL:           {kl_div:.6f}")
        if dropout_result:
            print(f"  MC Dropout ({mc_dropout_samples} samples):")
            print(f"    Avg predictive variance:      {dropout_result['variance_mean']:.6f}")
            print(f"    Accurate variance (mean):     {dropout_result['accurate_var_mean']:.6f}")
            print(f"    Inaccurate variance (mean):   {dropout_result['inaccurate_var_mean']:.6f}")
        print(f"{'─' * 70}\n")

    # ── Save plots ───────────────────────────────────────────────────
    if save_plots:
        plot_path = Path(plot_dir)
        plot_path.mkdir(parents=True, exist_ok=True)

        _plot_confusion_matrix(cm, class_names, model_name, plot_path)
        _plot_reliability_diagram(bin_conf, bin_acc, bin_counts, ece,
                                  model_name, plot_path)
        if dropout_result is not None and 'sample_variances' in dropout_result:
            _plot_mc_dropout_variance(dropout_result, model_name, plot_path)

    return result


if __name__ == "__main__":
    """Test the utility functions."""
    print("Testing my_utils module...")

    # Test get_device
    device = get_device()
    print(f"Device: {device}")

    # Test get_cifar10_data_augmentation
    transform_train, transform_test = get_cifar10_data_augmentation()
    print(f"Train transform: {transform_train}")
    print(f"Test transform: {transform_test}")

    print("All tests passed!")