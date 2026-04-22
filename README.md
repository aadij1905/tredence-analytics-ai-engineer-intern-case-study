# Self-Pruning Neural Network — CIFAR-10

**Tredence Analytics — AI Engineer Internship Case Study**  
**Author:** Aadi Jain  

📹 **[Video Explanation](https://drive.google.com/file/d/19x_OLbXyt8sOJJNcAVsEPRyqeKMsVwjg/view?usp=drive_link)**

---

## Overview

This project implements a neural network that **learns to prune itself** during training. Every weight is paired with a learnable gate parameter — an L1 penalty on these gates forces the network to identify and remove its own redundant parameters, producing a sparse model in a single training run.

Three pruning strategies are implemented and compared:

| Method | Type | Pruning Level |
|--------|------|---------------|
| **PrunableLinear** (Sigmoid + L1) | Unstructured (per-weight) | Baseline |
| **HardConcreteLinear** (L0) | Unstructured (per-weight) | Advanced |
| **PrunableConv2d** (Sigmoid + L1) | Structured (per-channel) | CNN Extension |

## Repository Structure

```
├── submission.py          # Single self-contained script (layers, models, training, visualization)
├── REPORT.md              # Detailed technical report with results and analysis
├── requirements.txt       # Python dependencies
├── gate_distributions.png             # Sigmoid+L1 gate histogram
├── gate_distributions_hardconcrete.png # HardConcrete+L0 gate histogram
└── gate_distributions_conv2d.png       # Conv2d structured pruning gate histogram
```

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the full experiment (trains all 3 methods × 3 λ values)
python3 submission.py
```

The script automatically detects GPU hardware (`cuda` / `mps`) and falls back to `cpu`.

## Key Results

| Method | λ | Accuracy | Sparsity |
|--------|---|----------|----------|
| Sigmoid+L1 | 1e-6 | 63.54% | 73.87% |
| Sigmoid+L1 | 1e-4 | 62.51% | 98.71% |
| HardConcrete+L0 | 1e-6 | 60.85% | 87.42% |
| PrunableConv2d | 1e-6 | **77.68%** | 0.60% |
| PrunableConv2d | 1e-4 | **77.19%** | 44.41% |

> The CNN with structured pruning achieves the highest accuracy while learning meaningful channel sparsity at higher λ values.

## Technical Highlights

- **Custom PyTorch layers** with differentiable gating mechanisms
- **Dual learning rates** — gate parameters use 2e-2 vs 1e-3 for weights (critical for Adam convergence)
- **Hard Concrete distribution** (Louizos et al., 2018) for principled L0 regularisation with exact zeros
- **Structured channel pruning** via `PrunableConv2d` for real-world inference speedups
