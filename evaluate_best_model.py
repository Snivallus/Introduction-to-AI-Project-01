"""Evaluate the best LeNet-5 tuned model (exp_53) to obtain Best F1 and Worst F1.

This is a one-shot remedial script: the Task 3 hyperparameter-tuning pipeline
did not record per-class F1 metrics for the best model.  Once the training
pipeline is updated to return richer experiment results, this file should be
removed (see the TODO in README.md).

Usage:
    python evaluate_best_model.py
"""

import torch
import torchvision
import torchvision.transforms as transforms
from pathlib import Path

from my_model import LeNet
from my_utils import (
    CIFAR_10_MEAN,
    CIFAR_10_STD,
    CIFAR_10_CLASS,
    get_device,
    evaluate,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
CHECKPOINT_DIR = Path("checkpoints/task3/exp_53")
BEST_MODEL_PATH = CHECKPOINT_DIR / "best_model.pth"
BATCH_SIZE = 256

# ---------------------------------------------------------------------------
# 1. Device
# ---------------------------------------------------------------------------
device = get_device()

# ---------------------------------------------------------------------------
# 2. Data
# ---------------------------------------------------------------------------
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(CIFAR_10_MEAN, CIFAR_10_STD),
])

testset = torchvision.datasets.CIFAR10(
    root="./dataset", train=False, download=False, transform=transform_test,
)
test_loader = torch.utils.data.DataLoader(
    testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=1,
)

# ---------------------------------------------------------------------------
# 3. Model
# ---------------------------------------------------------------------------
# Task 3 experiments used dropout=None (see the notebook cell where DROPOUT=None)
model = LeNet(dropout=None)
state_dict = torch.load(BEST_MODEL_PATH, map_location=device, weights_only=True)
model.load_state_dict(state_dict)
model = model.to(device)
model.eval()

print(f"Loaded checkpoint: {BEST_MODEL_PATH}")
print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

# ---------------------------------------------------------------------------
# 4. Evaluate (verbose=True to see the summary)
# ---------------------------------------------------------------------------
result = evaluate(
    model=model,
    test_loader=test_loader,
    device=device,
    model_name="LeNet-5_tuned_best_exp53",
    save_plots=False,
    skip_mc_dropout=True,       # model has no dropout layers
    verbose=True,
)

# ---------------------------------------------------------------------------
# 5. Extract Best / Worst F1
# ---------------------------------------------------------------------------
f1_scores = {
    cls: metrics["f1"]
    for cls, metrics in result["per_class_metrics"].items()
}
best_class = max(f1_scores, key=f1_scores.get)
worst_class = min(f1_scores, key=f1_scores.get)

print("\n" + "=" * 60)
print("  Best / Worst F1 — LeNet-5 tuned (best)")
print("=" * 60)
print(f"  Best F1:  {best_class:>8s} = {f1_scores[best_class]:.4f}")
print(f"  Worst F1: {worst_class:>8s} = {f1_scores[worst_class]:.4f}")
print("=" * 60)

# Print all F1 scores for completeness
print("\n  Per-class F1 scores:")
for cls_name in CIFAR_10_CLASS:
    print(f"    {cls_name:>10s}: {f1_scores[cls_name]:.4f}")
