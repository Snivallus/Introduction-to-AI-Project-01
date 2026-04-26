# CIFAR-10 Image Classification

This project systematically explores convolutional neural network (CNN) design and training on the CIFAR-10 benchmark. It covers the full workflow from baseline training to hyperparameter tuning and modern architecture implementation.

## Experimental Tasks

1. **LeNet-5 Baseline** — Train a classic LeNet-5 on CIFAR-10 and plot the training loss curve. Observe overfitting: a decreasing training loss does not guarantee high test accuracy, as the model memorises training patterns rather than learning generalisable features.

2. **Regularization** — Add L2 weight decay (via SGD's `weight_decay`) and Dropout to the LeNet-5 to mitigate overfitting. Compare training dynamics, test accuracy, confusion matrices, reliability diagrams, and MC Dropout uncertainty estimates against the baseline.

3. **Hyperparameter Tuning** — Perform a 4-factor grid search over learning rate (3 values), weight decay (3 values), label smoothing (2 values), and batch size (3 values), totalling 54 combinations. Analyse marginal effects and LR×BS interaction. Results are saved in `task3_hyperparameter_results.csv`.

4. **Modern CNN (MyCNN)** — Implement a ResNet-18 variant adapted for 32×32 inputs: replace the initial 7×7 convolution with a 3×3 convolution, remove the initial max-pooling, and adjust the final fully-connected layer for 10 classes. Train with stronger data augmentation and cosine annealing with warm restarts to surpass LeNet-5 performance.

## Requirements

- Python >= 3.11
- PyTorch
- torchvision
- numpy
- matplotlib

See `requirements.txt` for the full dependency list with version pins.

## Folder Structure

```
.
├── .gitattributes                   # Git attribute configuration
├── .gitignore                       # Git ignore rules
├── README.md                        # Project overview and file map (this file)
├── requirements.txt                 # Python package dependencies
├── project.ipynb                    # Main Jupyter notebook with all experiments
├── my_model.py                      # Model definitions: LeNet, BasicBlock, MyCNN
├── my_utils.py                      # Utility functions: training, evaluation, plotting
├── task3_hyperparameter_results.csv # Aggregated results from the 54-experiment grid search
│
├── checkpoints/                     # Model weight files (.pth) organised by experiment
│   ├── lenet_baseline/              #   Baseline LeNet-5 checkpoint
│   ├── lenet_dropout/               #   LeNet-5 with Dropout checkpoint
│   ├── task3/                       #   54 hyperparameter-tuning checkpoints (exp_01–exp_54)
│   └── mycnn/                       #   MyCNN (ResNet-18 variant) checkpoint
│
├── figures/                         # Generated evaluation plots (PDF)
│   ├── data_exploration_*.pdf       #   CIFAR-10 data exploration: class distribution, augmentation, etc.
│   ├── lenet_baseline_*.pdf         #   Task 1: loss curve, confusion matrix, reliability diagram
│   ├── lenet_dropout_*.pdf          #   Task 2: loss curve, confusion matrix, reliability diagram, etc.
│   ├── task3_*.pdf                  #   Task 3: hyperparameter effects, LR×BS interaction
│   └── mycnn_*.pdf                  #   Task 4: loss curve, confusion matrix, reliability diagram
│
├── latex/                           # LaTeX source for the project report
│   ├── neurips_2026.pdf             #   Compiled project report
│   ├── neurips_2026.tex             #   Main report (NeurIPS 2026 format)
│   ├── neurips_2026.sty             #   NeurIPS 2026 style file
│   └── references.bib               #   BibTeX bibliography
│
└── logs/                            # Training log files
    └── task3/                       #   54 log files named by hyperparameter combination
```
