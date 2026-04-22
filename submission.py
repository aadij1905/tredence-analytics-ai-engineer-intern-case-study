#!/usr/bin/env python3
"""
Self-Pruning Neural Network on CIFAR-10
========================================
Tredence Analytics — AI Engineer Internship Case Study

This script implements a neural network that **learns to prune itself**
during training.  Every weight in the network is paired with a learnable
"gate" parameter.  During the forward pass the gates (passed through
sigmoid) element-wise multiply the weights, so a gate near 0 effectively
removes its corresponding weight.  An L1 penalty on the gate values
encourages the optimiser to push unneeded gates to zero — producing a
sparse network in a single training run.

Two pruning strategies are implemented:
    A. PrunableLinear   — sigmoid gates + L1 regularisation (simple, stable)
    B. HardConcreteLinear — Hard Concrete gates + L0 regularisation (Louizos 2018)

Deliverables produced by running this script:
    1. Training logs for three λ values per method
    2. Markdown-formatted results table (Lambda | Accuracy | Sparsity %)
    3. gate_distributions.png — histogram of final gate values
    4. gate_distributions_hardconcrete.png — same for Hard Concrete method
    5. gate_distributions_conv2d.png — same for Conv2d structured pruning method

Author : Aadi Jain
Date   : April 2026
"""

# ── Imports ───────────────────────────────────────────────────────────
import math
from typing import List, Tuple

import matplotlib
matplotlib.use("Agg")                       # non-interactive backend (works headless)
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms


# ═══════════════════════════════════════════════════════════════════════
#  1.  PrunableLinear — The Core Custom Layer
# ═══════════════════════════════════════════════════════════════════════

class PrunableLinear(nn.Module):
    """
    Drop-in replacement for ``nn.Linear`` with learnable per-weight gates.

    For every entry w_ij in the weight matrix we maintain a learnable
    scalar ``gate_score_ij``.  The forward pass proceeds as:

        1.  gates = σ(gate_scores)           # values in (0, 1)
        2.  pruned_weights = weight ⊙ gates  # element-wise masking
        3.  output = x @ pruned_weights^T + bias

    When gate_score → −∞ the sigmoid → 0 and the corresponding weight
    is effectively removed.  Because the sigmoid is differentiable, the
    optimiser can learn which gates to close via back-propagation.

    Parameters
    ----------
    in_features  : int — size of each input sample
    out_features : int — size of each output sample
    bias         : bool — whether to include a bias term
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # ── Learnable weight + bias (standard linear layer parameters) ─
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias   = nn.Parameter(torch.zeros(out_features)) if bias else None

        # ── Learnable gate scores (same shape as weight) ──────────────
        # Initialised to 0 so that σ(0) = 0.5  → all gates start
        # "half-open", giving the optimiser freedom to push them in
        # either direction without initial bias.
        self.gate_scores = nn.Parameter(torch.zeros(out_features, in_features))

        # ── Weight initialisation (Kaiming uniform — nn.Linear default) ─
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with gated weights.

        The computation graph flows through the sigmoid so that
        ∂L/∂gate_scores is well-defined and the optimiser can
        update them.
        """
        # Step 1 — convert raw scores to [0, 1] gates
        gates = torch.sigmoid(self.gate_scores)

        # Step 2 — element-wise mask: if gate ≈ 0, weight is pruned
        pruned_weights = self.weight * gates

        # Step 3 — standard linear transform with masked weights
        return F.linear(x, pruned_weights, self.bias)

    # ── Utility helpers ───────────────────────────────────────────
    def get_gates(self) -> torch.Tensor:
        """Return the current gate values as a detached tensor."""
        with torch.no_grad():
            return torch.sigmoid(self.gate_scores)

    def sparsity_loss(self) -> torch.Tensor:
        """
        L1 norm of the sigmoid-activated gate values.

        This is the per-layer contribution to the sparsity penalty in
        the total loss:  L_total = CE + λ · Σ_layers |gates|₁
        """
        return torch.sigmoid(self.gate_scores).sum()

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}"


class PrunableConv2d(nn.Module):
    """
    Conv2d layer with learnable filter-level gates.

    Each output channel has one learnable gate score. The sigmoid-activated
    gate scales the whole filter, enabling structured channel pruning.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        bias: bool = True,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
        self.stride = stride
        self.padding = padding

        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, *self.kernel_size))
        self.bias = nn.Parameter(torch.zeros(out_channels)) if bias else None

        # One gate per output channel, broadcast over (in, kH, kW).
        self.gate_scores = nn.Parameter(torch.zeros(out_channels, 1, 1, 1))

        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gates = torch.sigmoid(self.gate_scores)
        pruned_weights = self.weight * gates
        return F.conv2d(x, pruned_weights, self.bias, self.stride, self.padding)

    def get_gates(self) -> torch.Tensor:
        with torch.no_grad():
            return torch.sigmoid(self.gate_scores)

    def sparsity_loss(self) -> torch.Tensor:
        return torch.sigmoid(self.gate_scores).sum()

    def extra_repr(self) -> str:
        return (
            f"in_channels={self.in_channels}, out_channels={self.out_channels}, "
            f"kernel_size={self.kernel_size}, stride={self.stride}, padding={self.padding}"
        )


# ═══════════════════════════════════════════════════════════════════════
#  1b. HardConcreteLinear — Advanced L0 Regularisation (Louizos 2018)
# ═══════════════════════════════════════════════════════════════════════

class HardConcreteLinear(nn.Module):
    """
    Linear layer with *Hard Concrete* gates (Louizos et al., 2018).

    Instead of sigmoid + L1, this layer samples binary-like masks from a
    stretched concrete distribution during training.  The expected L0 norm
    (the *number* of non-zero weights, not their magnitude) is differentiable
    and serves as the regulariser.

    Key advantages over PrunableLinear:
        • Produces **exactly zero** masks via hard clipping to [0, 1]
        • Regularises the *count* of active weights (L0), not their sum (L1)
        • No need to choose a post-hoc pruning threshold

    At inference time we use the deterministic mask:
        clamp(sigmoid(log_alpha) * (ζ − γ) + γ,  0, 1)

    Parameters
    ----------
    in_features, out_features, bias : same as ``nn.Linear``
    beta  : float — inverse temperature of the concrete distribution (default 2/3)
    gamma : float — stretch lower bound  (default −0.1)
    zeta  : float — stretch upper bound  (default  1.1)
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        beta: float = 2 / 3,
        gamma: float = -0.1,
        zeta: float = 1.1,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.beta = beta
        self.gamma = gamma
        self.zeta = zeta

        # ── Learnable parameters ──────────────────────────────────
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None

        # log_alpha controls the "openness" of each gate.
        # Initialised to 0 → sigmoid(0)=0.5 → gates start ~50% open.
        self.log_alpha = nn.Parameter(torch.zeros(out_features, in_features))

        # ── Weight initialisation ─────────────────────────────────
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def _sample_mask(self) -> torch.Tensor:
        """
        Sample from the Hard Concrete distribution (training only).

        u ~ Uniform(ε, 1−ε)
        s = sigmoid((log(u) − log(1−u) + log_alpha) / β)
        s_bar = s * (ζ − γ) + γ
        mask = clamp(s_bar, 0, 1)   ← this clamp creates exact zeros
        """
        u = torch.zeros_like(self.log_alpha).uniform_(1e-8, 1 - 1e-8)
        s = torch.sigmoid(
            (torch.log(u) - torch.log(1 - u) + self.log_alpha) / self.beta
        )
        s_bar = s * (self.zeta - self.gamma) + self.gamma
        return s_bar.clamp(0.0, 1.0)

    def _deterministic_mask(self) -> torch.Tensor:
        """Deterministic mask for inference (no sampling)."""
        return (
            torch.sigmoid(self.log_alpha)
            * (self.zeta - self.gamma)
            + self.gamma
        ).clamp(0.0, 1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mask = self._sample_mask() if self.training else self._deterministic_mask()
        pruned_weights = self.weight * mask
        return F.linear(x, pruned_weights, self.bias)

    # ── Regularisation & utilities ────────────────────────────────
    def sparsity_loss(self) -> torch.Tensor:
        """
        Expected L0 norm — the probability each gate is non-zero.

        This is the CDF of the Hard Concrete distribution evaluated
        at zero, summed over all gates.  It counts *how many* weights
        are expected to be active, which is a more principled measure
        than L1.
        """
        return torch.sigmoid(
            self.log_alpha - self.beta * math.log(-self.gamma / self.zeta)
        ).sum()

    def get_gates(self) -> torch.Tensor:
        """Return the deterministic gate values as a detached tensor."""
        return self._deterministic_mask().detach()

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"beta={self.beta}, gamma={self.gamma}, zeta={self.zeta}"
        )


# ═══════════════════════════════════════════════════════════════════════
#  2.  SelfPruningNet — CIFAR-10 Classifier
# ═══════════════════════════════════════════════════════════════════════

class SelfPruningNet(nn.Module):
    """
    Feed-forward classifier built entirely from PrunableLinear layers.

    Architecture
    ------------
        Flatten(3×32×32 = 3072)
          → PrunableLinear(3072, 512)  → BatchNorm → ReLU
          → PrunableLinear( 512, 256)  → BatchNorm → ReLU
          → PrunableLinear( 256, 128)  → BatchNorm → ReLU
          → PrunableLinear( 128,  10)  → (logits, no activation)

    Total learnable parameters ≈ 1.86 M (half are gate scores).
    """

    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()

        # Every linear layer is prunable
        self.layers = nn.ModuleList([
            PrunableLinear(3 * 32 * 32, 512),   # 3072 → 512
            PrunableLinear(512, 256),            #  512 → 256
            PrunableLinear(256, 128),            #  256 → 128
            PrunableLinear(128,  10),            #  128 →  10  (logits)
        ])

        # BatchNorm after each hidden layer (not the final logit layer)
        self.bns = nn.ModuleList([
            nn.BatchNorm1d(512),
            nn.BatchNorm1d(256),
            nn.BatchNorm1d(128),
        ])
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.flatten(x)                     # (B, 3, 32, 32) → (B, 3072)
        for i, layer in enumerate(self.layers[:-1]):
            x = layer(x)                        # PrunableLinear
            x = self.bns[i](x)                  # BatchNorm
            x = self.relu(x)                    # ReLU
        return self.layers[-1](x)               # final layer → raw logits

    # ── Gate aggregation helpers ──────────────────────────────────
    def total_sparsity_loss(self) -> torch.Tensor:
        """Sum of L1(gates) across all PrunableLinear layers."""
        return sum(layer.sparsity_loss() for layer in self.layers)

    def all_gates(self) -> torch.Tensor:
        """Concatenate all gate values into a single flat tensor."""
        return torch.cat([layer.get_gates().flatten() for layer in self.layers])

    def sparsity_level(self, threshold: float = 1e-2) -> float:
        """Fraction of gates below ``threshold`` (= effectively pruned)."""
        gates = self.all_gates()
        return (gates < threshold).float().mean().item()


class HardConcretePruningNet(nn.Module):
    """
    Same architecture as SelfPruningNet but using HardConcreteLinear.

    This provides a direct comparison between the two gating methods
    while keeping everything else identical.
    """

    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()

        self.layers = nn.ModuleList([
            HardConcreteLinear(3 * 32 * 32, 512),
            HardConcreteLinear(512, 256),
            HardConcreteLinear(256, 128),
            HardConcreteLinear(128, 10),
        ])

        self.bns = nn.ModuleList([
            nn.BatchNorm1d(512),
            nn.BatchNorm1d(256),
            nn.BatchNorm1d(128),
        ])
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.flatten(x)
        for i, layer in enumerate(self.layers[:-1]):
            x = layer(x)
            x = self.bns[i](x)
            x = self.relu(x)
        return self.layers[-1](x)

    def total_sparsity_loss(self) -> torch.Tensor:
        """Sum of L0(gates) across all HardConcreteLinear layers."""
        return sum(layer.sparsity_loss() for layer in self.layers)

    def all_gates(self) -> torch.Tensor:
        return torch.cat([layer.get_gates().flatten() for layer in self.layers])

    def sparsity_level(self, threshold: float = 1e-2) -> float:
        gates = self.all_gates()
        return (gates < threshold).float().mean().item()


class ConvPruningNet(nn.Module):
    """
    CIFAR-10 classifier with structured-prunable Conv2d layers.

    Architecture
    ------------
      Conv(3->32, k3) -> BN -> ReLU -> MaxPool
      Conv(32->64, k3) -> BN -> ReLU -> MaxPool
      Conv(64->128, k3) -> BN -> ReLU
      AdaptiveAvgPool(1x1)
      PrunableLinear(128->10)
    """

    def __init__(self):
        super().__init__()
        self.conv_layers = nn.ModuleList([
            PrunableConv2d(3, 32, kernel_size=3, padding=1),
            PrunableConv2d(32, 64, kernel_size=3, padding=1),
            PrunableConv2d(64, 128, kernel_size=3, padding=1),
        ])
        self.conv_bns = nn.ModuleList([
            nn.BatchNorm2d(32),
            nn.BatchNorm2d(64),
            nn.BatchNorm2d(128),
        ])
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = PrunableLinear(128, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.conv_bns[0](self.conv_layers[0](x)))
        x = self.pool(x)

        x = self.relu(self.conv_bns[1](self.conv_layers[1](x)))
        x = self.pool(x)

        x = self.relu(self.conv_bns[2](self.conv_layers[2](x)))
        x = self.avg_pool(x)
        x = x.flatten(1)
        return self.classifier(x)

    def total_sparsity_loss(self) -> torch.Tensor:
        conv_loss = sum(layer.sparsity_loss() for layer in self.conv_layers)
        return conv_loss + self.classifier.sparsity_loss()

    def all_gates(self) -> torch.Tensor:
        conv_gates = [layer.get_gates().flatten() for layer in self.conv_layers]
        return torch.cat(conv_gates + [self.classifier.get_gates().flatten()])

    def sparsity_level(self, threshold: float = 1e-2) -> float:
        gates = self.all_gates()
        return (gates < threshold).float().mean().item()


# ═══════════════════════════════════════════════════════════════════════
#  3.  Data Loading — CIFAR-10
# ═══════════════════════════════════════════════════════════════════════

def get_cifar10_loaders(batch_size: int = 256) -> Tuple:
    """
    Download CIFAR-10 (if needed) and return train / test DataLoaders.

    Train-time augmentation: random horizontal flip + random crop.
    Both splits are normalised with per-channel mean/std.
    """
    mean = (0.4914, 0.4822, 0.4465)
    std  = (0.2023, 0.1994, 0.2010)

    train_tf = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    test_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    train_set = torchvision.datasets.CIFAR10(
        root="./data", train=True,  download=True, transform=train_tf)
    test_set  = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=True, transform=test_tf)

    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=True,
        num_workers=0, pin_memory=True, drop_last=True)
    test_loader  = torch.utils.data.DataLoader(
        test_set,  batch_size=batch_size, shuffle=False,
        num_workers=0, pin_memory=True)

    return train_loader, test_loader


# ═══════════════════════════════════════════════════════════════════════
#  4.  Training & Evaluation
# ═══════════════════════════════════════════════════════════════════════

def train_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    lam: float,
    device: torch.device,
) -> Tuple[float, float]:
    """
    Train for one epoch with the composite loss:

        L = CrossEntropy(logits, labels) + λ · Σ|gates|

    Returns (avg_cls_loss, avg_total_loss).
    """
    model.train()
    cls_sum = total_sum = 0.0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad(set_to_none=True)   # slightly faster than zero_grad()

        logits = model(images)

        # ── Composite loss ─────────────────────────────────────────
        cls_loss      = criterion(logits, labels)          # classification
        sparsity_loss = lam * model.total_sparsity_loss()  # L1 on gates
        loss          = cls_loss + sparsity_loss

        loss.backward()

        # Gradient clipping — prevents instability from L1 term
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        cls_sum   += cls_loss.item()
        total_sum += loss.item()

    n = len(loader)
    return cls_sum / n, total_sum / n


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> float:
    """Return test accuracy as a fraction in [0, 1]."""
    model.eval()
    correct = total = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        preds = model(images).argmax(dim=1)
        correct += (preds == labels).sum().item()
        total   += labels.size(0)
    return correct / total


def train_and_evaluate(
    lam: float,
    epochs: int = 50,
    device: torch.device = None,
    method: str = "sigmoid",
) -> dict:
    """
    Full training run for a given λ (sparsity strength).

    Parameters
    ----------
    lam    : regularisation strength
    epochs : number of training epochs
    device : torch device
    method : "sigmoid" for PrunableLinear (L1),
             "hardconcrete" for HardConcreteLinear (L0), or
             "conv2d" for PrunableConv2d + PrunableLinear (L1)

    Returns
    -------
    dict with keys:
        lam, method, test_accuracy (%), sparsity (%), model, gate_values
    """
    if device is None:
        device = torch.device("cpu")

    if method == "sigmoid":
        label = "Sigmoid+L1"
    elif method == "hardconcrete":
        label = "HardConcrete+L0"
    elif method == "conv2d":
        label = "PrunableConv2d+L1"
    else:
        raise ValueError("method must be one of: sigmoid, hardconcrete, conv2d")
    print(f"\n{'='*60}")
    print(f"  [{label}]  λ = {lam}  |  device = {device}")
    print(f"{'='*60}")

    # ── Data ──────────────────────────────────────────────────────
    train_loader, test_loader = get_cifar10_loaders(batch_size=256)

    # ── Model + optimiser ─────────────────────────────────────────
    if method == "hardconcrete":
        model = HardConcretePruningNet().to(device)
        # Gate params are "log_alpha" in HardConcreteLinear
        gate_params = [p for n, p in model.named_parameters() if "log_alpha" in n]
        base_params = [p for n, p in model.named_parameters() if "log_alpha" not in n]
    elif method == "conv2d":
        model = ConvPruningNet().to(device)
        gate_params = [p for n, p in model.named_parameters() if "gate" in n]
        base_params = [p for n, p in model.named_parameters() if "gate" not in n]
    else:
        model = SelfPruningNet().to(device)
        gate_params = [p for n, p in model.named_parameters() if "gate" in n]
        base_params = [p for n, p in model.named_parameters() if "gate" not in n]

    optimizer = optim.Adam([
        {"params": base_params, "weight_decay": 1e-4},
        {"params": gate_params, "lr": 2e-2, "weight_decay": 0.0}
    ], lr=1e-3)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()

    # ── Training loop ─────────────────────────────────────────────
    for epoch in range(1, epochs + 1):
        cls_loss, total_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, lam, device)
        scheduler.step()

        # Log every 5 epochs + first and last
        if epoch % 5 == 0 or epoch == 1 or epoch == epochs:
            acc      = evaluate(model, test_loader, device)
            sparsity = model.sparsity_level()
            print(
                f"  Epoch {epoch:02d}/{epochs} │ "
                f"Cls Loss: {cls_loss:.4f} │ Total Loss: {total_loss:.4f} │ "
                f"Acc: {acc*100:.2f}% │ Sparsity: {sparsity*100:.1f}%"
            )

    # ── Final metrics ─────────────────────────────────────────────
    final_acc      = evaluate(model, test_loader, device)
    final_sparsity = model.sparsity_level()
    gate_values    = model.all_gates().cpu()

    print(f"\n  ✓ Final Test Accuracy : {final_acc*100:.2f}%")
    print(f"  ✓ Final Sparsity Level: {final_sparsity*100:.2f}%")

    return {
        "lam":           lam,
        "method":        label,
        "test_accuracy": final_acc * 100,
        "sparsity":      final_sparsity * 100,
        "model":         model,
        "gate_values":   gate_values,
    }


# ═══════════════════════════════════════════════════════════════════════
#  5.  Visualisation & Reporting
# ═══════════════════════════════════════════════════════════════════════

def plot_gate_distributions(results: List[dict], save_path: str = "gate_distributions.png"):
    """
    Side-by-side histograms of gate values for each λ experiment.

    A successful run shows a bimodal distribution:
        • spike near 0  → pruned weights
        • cluster > 0   → retained weights
    """
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4.5))
    if n == 1:
        axes = [axes]

    best = max(results, key=lambda r: r["test_accuracy"])
    colours = ["#f38ba8", "#a6e3a1", "#89b4fa"]

    for ax, result, colour in zip(axes, results, colours):
        gates = result["gate_values"].numpy()
        ax.hist(gates, bins=60, color=colour, edgecolor="white", alpha=0.85, linewidth=0.5)

        # Mark the pruning threshold
        ax.axvline(0.01, color="red", ls="--", lw=1, alpha=0.7, label="threshold (0.01)")

        ax.set_xlabel("Gate Value", fontsize=11)
        ax.set_ylabel("Count", fontsize=11)
        ax.set_xlim(-0.02, 1.02)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc="upper center")

        star = " ★ Best" if result["lam"] == best["lam"] else ""
        ax.set_title(
            f"λ = {result['lam']:.0e}{star}\n"
            f"Acc = {result['test_accuracy']:.2f}%  |  "
            f"Sparsity = {result['sparsity']:.1f}%",
            fontsize=10,
        )

    fig.suptitle(
        "Self-Pruning Neural Network — Gate Value Distributions\n"
        "(spike near 0 → pruned weights  |  values near 1 → retained weights)",
        fontsize=12, fontweight="bold", y=1.04,
    )
    plt.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    print(f"\n  📊 Plot saved → {save_path}")
    return fig


def print_results_table(results: List[dict], title: str = "Results Summary") -> None:
    """Print a Markdown-formatted results comparison table."""
    print(f"\n\n## {title}\n")
    print(f"| {'Method':<18} | {'Lambda':>10} | {'Test Accuracy (%)':>18} | {'Sparsity Level (%)':>20} |")
    print(f"|{'-'*20}|{'-'*12}|{'-'*20}|{'-'*22}|")
    for r in results:
        print(
            f"| {r.get('method', 'Sigmoid+L1'):<18} "
            f"| {r['lam']:>10.0e} "
            f"| {r['test_accuracy']:>18.2f} "
            f"| {r['sparsity']:>20.2f} |"
        )
    print()


# ═══════════════════════════════════════════════════════════════════════
#  6.  Main — Run the Full Experiment
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # ── Reproducibility ───────────────────────────────────────────
    torch.manual_seed(42)
    np.random.seed(42)

    # ── Device selection ──────────────────────────────────────────
    if torch.cuda.is_available():
        DEVICE = torch.device("cuda")
    elif torch.backends.mps.is_available():
        DEVICE = torch.device("mps")
    else:
        DEVICE = torch.device("cpu")

    # ── Experiment config ─────────────────────────────────────────
    EPOCHS  = 50                           # increase to 30+ for better accuracy
    LAMBDAS = [1e-6, 1e-5, 1e-4]          # low / medium / high sparsity pressure

    # ══════════════════════════════════════════════════════════════
    #  Method A — Sigmoid + L1 (core method)
    # ══════════════════════════════════════════════════════════════
    print("\n" + "#" * 60)
    print("#  METHOD A: PrunableLinear (Sigmoid + L1)")
    print("#" * 60)

    sigmoid_results = []
    for lam in LAMBDAS:
        torch.manual_seed(42)              # re-seed for fair comparison
        result = train_and_evaluate(
            lam=lam, epochs=EPOCHS, device=DEVICE, method="sigmoid")
        sigmoid_results.append(result)

    print_results_table(sigmoid_results, title="Sigmoid + L1 Results")
    plot_gate_distributions(sigmoid_results, save_path="gate_distributions.png")

    # ══════════════════════════════════════════════════════════════
    #  Method B — Hard Concrete + L0 (advanced extension)
    # ══════════════════════════════════════════════════════════════
    print("\n" + "#" * 60)
    print("#  METHOD B: HardConcreteLinear (L0 Regularisation)")
    print("#" * 60)

    hc_results = []
    for lam in LAMBDAS:
        torch.manual_seed(42)
        result = train_and_evaluate(
            lam=lam, epochs=EPOCHS, device=DEVICE, method="hardconcrete")
        hc_results.append(result)

    print_results_table(hc_results, title="Hard Concrete + L0 Results")
    plot_gate_distributions(hc_results, save_path="gate_distributions_hardconcrete.png")

    # ══════════════════════════════════════════════════════════════
    #  Method C — PrunableConv2d + L1 (structured pruning)
    # ══════════════════════════════════════════════════════════════
    print("\n" + "#" * 60)
    print("#  METHOD C: PrunableConv2d (Structured Sigmoid + L1)")
    print("#" * 60)

    conv2d_results = []
    for lam in LAMBDAS:
        torch.manual_seed(42)
        result = train_and_evaluate(
            lam=lam, epochs=EPOCHS, device=DEVICE, method="conv2d")
        conv2d_results.append(result)

    print_results_table(conv2d_results, title="PrunableConv2d + L1 Results")
    plot_gate_distributions(conv2d_results, save_path="gate_distributions_conv2d.png")

    # ══════════════════════════════════════════════════════════════
    #  Combined comparison
    # ══════════════════════════════════════════════════════════════
    print_results_table(
        sigmoid_results + hc_results + conv2d_results,
        title="Combined Comparison: Sigmoid+L1 vs HardConcrete+L0 vs PrunableConv2d+L1"
    )
