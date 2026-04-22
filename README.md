# Self-Pruning Neural Network

This repository contains my submission for the Tredence Analytics AI Engineer Internship Case Study. It implements a neural network in PyTorch that dynamically learns to prune itself during training, identifying and removing redundant parameters via learnable sigmoid gates and an L1 sparsity penalty.

## Repository Structure

- `submission.py`: The single executable Python script containing the custom PyTorch layer implementations (`PrunableLinear`, `HardConcreteLinear`, `PrunableConv2d`), the neural network architectures, and the complete training/evaluation loop.
- `REPORT.md`: The required write-up detailing the mathematical mechanism behind the L1 penalty, summarizing the empirical results, and providing an analysis of the accuracy-sparsity trade-off.
- `requirements.txt`: The Python dependencies needed to run the code.
- `gate_distributions*.png`: Visualizations of the final gate values for the different pruning methods (generated automatically by running the script).

## Setup & Execution

### 1. Install Dependencies
Ensure you have Python 3 installed, then install the required packages:

```bash
pip install -r requirements.txt
```

*(Note: PyTorch is included in the requirements, but depending on your hardware (CUDA/MPS), you may want to install the hardware-specific PyTorch binaries from the [official website](https://pytorch.org/get-started/locally/).)*

### 2. Run the Experiment
To execute the training loop, evaluate the models, and automatically generate the results tables and plots, simply run:

```bash
python3 submission.py
```

The script is configured to run 50 epochs across three different sparsity penalty strengths (`λ = [1e-6, 1e-5, 1e-4]`) for three different pruning architectures. It will automatically detect and utilize GPU hardware (`cuda` or `mps`) if available, falling back to `cpu` otherwise.
