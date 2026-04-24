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

# Standard library
from typing import Dict, List, Tuple, Optional, Any

# Third-party
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from torchvision.transforms import RandomApply
import matplotlib.pyplot as plt
from pathlib import Path


CIFAR_10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_10_STD = (0.2470, 0.2435, 0.2616)
CIFAR_10_CLASS = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


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
        transform_train = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.1),
            transforms.RandomVerticalFlip(p=0.1),
            RandomApply([transforms.RandomCrop(32, padding=4)], p=0.1),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR_10_MEAN, CIFAR_10_STD),
        ])
    elif style == 'full':
        # Aggressive augmentation suitable for large-capacity models.
        transform_train = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            RandomApply([transforms.RandomCrop(32, padding=4)], p=0.5),
            RandomApply([transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)], p=0.5),
            RandomApply([transforms.RandomRotation(30)], p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR_10_MEAN, CIFAR_10_STD),
            transforms.RandomErasing(p=0.5, scale=(0.02, 0.2), ratio=(0.3, 3.3), value=0),
            RandomApply([transforms.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 1.0))], p=0.5)
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
    restart_count = 0

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
                old_lr = optimizer.param_groups[0]['lr']
                scheduler.step()
                new_lr = optimizer.param_groups[0]['lr']
                if new_lr > old_lr * 1.5:
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
            if i % print_every == print_every - 1:
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
        'total': total_params,
        'trainable': trainable_params,
        'non_trainable': non_trainable_params,
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
    if count >= 1e9:
        return f"{count / 1e9:.{decimals}f}B".rstrip('0').rstrip('.') + 'B'
    elif count >= 1e6:
        return f"{count / 1e6:.{decimals}f}M".rstrip('0').rstrip('.') + 'M'
    elif count >= 1e3:
        return f"{count / 1e3:.{decimals}f}K".rstrip('0').rstrip('.') + 'K'
    else:
        return str(count)


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
            min(5, max(2, 10% of total_epochs)).
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
        warmup_epochs = min(5, max(2, int(total_epochs * 0.1)))
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


# ═══════════════════════════════════════════════════════════════════════
# Comprehensive Evaluation: Beyond Top-1 Accuracy
# ═══════════════════════════════════════════════════════════════════════


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

    try:
        from sklearn.metrics import precision_recall_fscore_support
        prec, rec, f1, _ = precision_recall_fscore_support(
            targets, predictions, labels=range(num_classes), zero_division=0
        )
    except ImportError:
        prec = np.zeros(num_classes)
        rec = np.zeros(num_classes)
        f1 = np.zeros(num_classes)
        for i in range(num_classes):
            tp = ((predictions == i) & (targets == i)).sum()
            fp = ((predictions == i) & (targets != i)).sum()
            fn = ((predictions != i) & (targets == i)).sum()
            prec[i] = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec[i] = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1[i] = (
                2 * prec[i] * rec[i] / (prec[i] + rec[i])
                if (prec[i] + rec[i]) > 0
                else 0.0
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
# 5. CIFAR-10-C corruption robustness
# ------------------------------------------------------------------

def evaluate_cifar10c(
    model: nn.Module,
    device: torch.device,
    data_root: str = './dataset',
    corruptions: Optional[List[str]] = None,
    severity: int = 5,
) -> Optional[Dict[str, float]]:
    """Evaluate corruption robustness on CIFAR-10-C.

    CIFAR-10-C must be downloaded separately from:
    https://github.com/hendrycks/robustness

    The dataset contains 19 corruptions at 5 severity levels.  By default
    this function evaluates on a representative subset of 4 corruptions
    at the hardest severity level.

    Args:
        model: PyTorch model.
        device: Computation device.
        data_root: Root directory containing ``CIFAR-10-C/``.
        corruptions: Corruption names to evaluate.  If ``None``, uses
            ``['gaussian_noise', 'defocus_blur', 'contrast',
            'elastic_transform']``.
        severity: Corruption severity (1 = mildest, 5 = hardest).

    Returns:
        Dict mapping corruption name → accuracy percentage, or ``None``
        if the ``CIFAR-10-C`` directory is not found.
    """

    cifar10c_root = Path(data_root) / 'CIFAR-10-C'
    if not cifar10c_root.is_dir():
        return None

    if corruptions is None:
        corruptions = [
            'gaussian_noise',
            'defocus_blur',
            'contrast',
            'elastic_transform',
        ]

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_10_MEAN, CIFAR_10_STD),
    ])

    labels = np.load(str(cifar10c_root / 'labels.npy'))
    results: Dict[str, float] = {}
    model.eval()

    for corruption in corruptions:
        data_path = cifar10c_root / f'{corruption}.npy'
        if not data_path.exists():
            print(f"  Warning: {corruption}.npy not found, skipping.")
            continue

        images = np.load(str(data_path))
        start = (severity - 1) * 10000
        end = severity * 10000
        images_sev = images[start:end]
        labels_sev = labels[start:end]

        # Inline dataset for this corruption
        class _CIFAR10CDataset(Dataset):
            def __init__(self, imgs: np.ndarray, lbls: np.ndarray,
                         xform: transforms.Compose) -> None:
                self.images = imgs
                self.labels = lbls
                self.transform = xform

            def __len__(self) -> int:
                return len(self.images)

            def __getitem__(self, idx: int):
                img = self.transform(self.images[idx])
                return img, int(self.labels[idx])

        dataset = _CIFAR10CDataset(images_sev, labels_sev, transform)
        loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=2)
        results[corruption] = evaluate_accuracy(model, loader, device)

    return results


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
    compute_cifar10c: bool = False,
    cifar10c_root: str = './dataset',
    verbose: bool = True,
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
    5. **CIFAR-10-C corruption robustness** (via :func:`evaluate_cifar10c`;
       only when ``compute_cifar10c=True`` and data is found).

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
            Dropout (ignored if the model has no dropout).
        compute_cifar10c: If ``True``, attempt CIFAR-10-C evaluation.
        cifar10c_root: Parent directory of ``CIFAR-10-C/``.
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
        - ``cifar10c`` — dict from :func:`evaluate_cifar10c`, or ``None``
          if not requested or data unavailable
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
    if _has_dropout(model):
        if verbose:
            print("\n[MC Dropout] Evaluating uncertainty ...")
        dropout_result = mc_dropout_evaluate(
            model, test_loader, device, num_samples=mc_dropout_samples
        )
        # Strip large arrays from the result dict unless plots are requested
        if not save_plots:
            dropout_result.pop('sample_variances', None)
            dropout_result.pop('accurate_mask', None)

    # ── 9. CIFAR-10-C ────────────────────────────────────────────────
    cifar10c_result: Optional[Dict[str, float]] = None
    if compute_cifar10c:
        if verbose:
            print("\n[CIFAR-10-C] Evaluating corruption robustness ...")
        cifar10c_result = evaluate_cifar10c(model, device, data_root=cifar10c_root)
        if cifar10c_result is None and verbose:
            print("  → CIFAR-10-C not found.  Download from "
                  "https://github.com/hendrycks/robustness and place in "
                  "``./dataset/CIFAR-10-C/``.")

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
        'cifar10c': cifar10c_result,
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
        print(f"  Pred distribution KL:            {kl_div:.6f}")
        if dropout_result:
            print(f"  MC Dropout ({mc_dropout_samples} samples):")
            print(f"    Avg predictive variance:      {dropout_result['variance_mean']:.6f}")
            print(f"    Accurate variance (mean):     {dropout_result['accurate_var_mean']:.6f}")
            print(f"    Inaccurate variance (mean):   {dropout_result['inaccurate_var_mean']:.6f}")
        if cifar10c_result:
            avg_c = float(np.mean(list(cifar10c_result.values())))
            print(f"  CIFAR-10-C (avg {len(cifar10c_result)} corruptions): {avg_c:.2f}%")
            for corr_name, corr_acc in cifar10c_result.items():
                drop = accuracy - corr_acc
                print(f"    {corr_name:30s}: {corr_acc:.2f}%  (Δ = {drop:.2f}%)")
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