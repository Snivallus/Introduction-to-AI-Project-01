"""Utility functions for CIFAR-10 image classification project.

This module contains common utilities for training, evaluation, and visualization
of neural networks on the CIFAR-10 dataset. Functions are type-annotated and
follow the project's coding style guidelines.

Example:
    >>> from my_utils import get_device, plot_loss_curves
    >>> device = get_device()
    >>> print(f"Using device: {device}")
"""

# Standard library
import os
from typing import Dict, List, Tuple, Optional, Union, Any

# Third-party
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np


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


def get_cifar10_data_augmentation() -> Tuple[transforms.Compose, transforms.Compose]:
    """Get data augmentation transforms for CIFAR-10.

    Returns:
        Tuple of (transform_train, transform_test) where:
        - transform_train: Transformations for training data (with augmentation)
        - transform_test: Transformations for test data (basic normalization)

    Note:
        Training augmentations include: random horizontal flip, random crop,
        color jitter, random rotation, and random erasing.
    """
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomCrop(32, padding=4),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        transforms.RandomErasing(p=0.5, scale=(0.02, 0.2), ratio=(0.3, 3.3), value=0)
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    return transform_train, transform_test


def create_train_val_split(
    dataset: Dataset,
    val_ratio: float = 0.1,
    seed: int = 42
) -> Tuple[Dataset, Dataset]:
    """Split dataset into training and validation sets.

    Args:
        dataset: PyTorch Dataset to split.
        val_ratio: Proportion of data to use for validation (0.0 to 1.0).
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (train_dataset, val_dataset).

    Raises:
        ValueError: If val_ratio is not between 0 and 1.
    """
    if not 0 <= val_ratio <= 1:
        raise ValueError(f"val_ratio must be between 0 and 1, got {val_ratio}")

    # Set random seed for reproducibility
    torch.manual_seed(seed)

    val_size = int(val_ratio * len(dataset))
    train_size = len(dataset) - val_size

    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )

    print(f"Train set size: {train_size}, Validation set size: {val_size}")
    return train_dataset, val_dataset


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
    print_every: int = 1000,
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
        early_stopping_patience: Number of epochs to wait for improvement
            before early stopping (optional).
        gradient_clip: Maximum gradient norm for clipping (optional).
        print_every: Print training status every N batches.
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

    device = next(model.parameters()).device
    print(f"Training on device: {device}")

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
                # Step the scheduler once per epoch after warm-up
                scheduler.step()

        running_loss = 0.0
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

            # 5. Gradient clipping (if specified)
            if gradient_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=gradient_clip)

            # 6. Optimizer step
            optimizer.step()

            # Record loss
            running_loss += loss.item()
            total_batches += 1

            # Print training status
            if i % print_every == print_every - 1:
                avg_loss = running_loss / total_batches
                print(f'Epoch {epoch+1}: batch {i+1:5d} loss: {avg_loss:.3f}')

        # Calculate average training loss for the epoch
        epoch_avg_loss = running_loss / total_batches if total_batches > 0 else 0
        train_losses.append(epoch_avg_loss)

        # Validation phase (if validation loader provided)
        if val_loader is not None:
            val_accuracy = evaluate_accuracy(model, val_loader, device)
            val_accuracies.append(val_accuracy)  # type: ignore

            print(f'Epoch {epoch+1}/{num_epochs}: '
                  f'Train Loss: {epoch_avg_loss:.4f}, '
                  f'Val Accuracy: {val_accuracy:.2f}%')

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
            print(f'Epoch {epoch+1}/{num_epochs}: Train Loss: {epoch_avg_loss:.4f}')

        # Save checkpoint for this epoch
        torch.save(model.state_dict(), f"{save_path}/epoch_{epoch + 1}_model.pth")

    print('Finished Training')

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


def predict(
    model: nn.Module,
    test_loader: DataLoader,
    device: Optional[torch.device] = None
) -> float:
    """Predict and calculate accuracy on test set.

    Args:
        model: PyTorch model to evaluate.
        test_loader: DataLoader for test data.
        device: Device to evaluate on (if None, uses model's device).

    Returns:
        Test accuracy percentage (0-100).

    Example:
        >>> test_accuracy = predict(model, test_loader)
        >>> print(f'测试集中的准确率为: {test_accuracy:.2f} %')
    """
    accuracy = evaluate_accuracy(model, test_loader, device)
    print(f'测试集中的准确率为: {accuracy:.2f} %')
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


def count_parameters(model: nn.Module) -> Dict[str, int]:
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

    return {
        'total': total_params,
        'trainable': trainable_params
    }


def create_learning_rate_scheduler(
    optimizer: optim.Optimizer,
    scheduler_type: str = 'cosine',
    total_epochs: int = 20,
    warmup_epochs: Optional[int] = None,
    initial_lr: float = 0.001,
    min_lr: float = 1e-6,
    T_0: Optional[int] = None,
    T_mult: int = 2
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
        warmup_epochs: Number of warm-up epochs. If None, defaults to
            max(2, 5% of total_epochs).
        initial_lr: Initial learning rate (used to scale warm-up).
        min_lr: Minimum learning rate (eta_min).
        T_0: Period for first restart in epochs. If None, defaults to
            total_epochs // 2.
        T_mult: Multiplication factor for T_0 after each restart.

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
        warmup_epochs = max(2, int(total_epochs * 0.05))
    if T_0 is None:
        T_0 = max(1, total_epochs // 2)

    # Suppress the "was invoked with deprecated argument" spam from PyTorch
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        if scheduler_type == 'cosine':
            from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
            scheduler = CosineAnnealingWarmRestarts(
                optimizer, T_0=T_0, T_mult=T_mult, eta_min=min_lr
            )
            scheduler_desc = (
                f"CosineAnnealingWarmRestarts (T_0={T_0}, T_mult={T_mult})"
            )
        elif scheduler_type == 'step':
            from torch.optim.lr_scheduler import StepLR
            step_size = max(1, total_epochs // 3)
            scheduler = StepLR(optimizer, step_size=step_size, gamma=0.1)
            scheduler_desc = f"StepLR (step_size={step_size}, gamma=0.1)"
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