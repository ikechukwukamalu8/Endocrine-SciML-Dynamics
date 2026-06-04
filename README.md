# Physics-Informed Neural Networks (PINNs) for Inverse Modelling and Parameter Discovery in Endocrine Dynamics

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![Framework: PyTorch](https://img.shields.io/badge/Framework-PyTorch-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 📌 Project Overview
This repository implements a **Physics-Informed Neural Network (PINN)** to solve the inverse problem of personalized parameter discovery in metabolic systems biology. Standard data-driven machine learning models are heavily prone to unphysiological anomalies when trained on sparse, noisy clinical datasets. 

By embedding the non-linear **Bergman Minimal Model (ODEs)** directly into the neural network's backpropagation loss function, this framework seamlessly uncovers hidden, patient-specific biological parameters—such as the insulin sensitivity/pancreatic supply metric ($p_3$)—from sparse clinical glucose trajectories.

## 🧬 Problem Statement & Mathematical Framework
In digital endocrinology, real-time monitors capture highly erratic and sparse time-series configurations. Traditional deep learning architectures cannot guarantee conservation of mass or follow biological constraints. 

This project bridges data-driven optimization with mechanistic dynamic simulation using a custom multi-objective loss landscape. The underlying physiological system state is governed by a modified set of differential equations tracking blood glucose concentration $G(t)$ and remote insulin action activity $X(t)$:

$$\frac{dG}{dt} = -p_1(G(t) - G_b) - X(t)G(t)$$

$$\frac{dX}{dt} = -p_2X(t) + p_3 \cdot f(t)$$

### The Multi-Objective Loss Design
The network minimizes a dual-component objective function during optimization:
1. **Data Loss ($\mathcal{L}_{\text{data}}$):** Mean Squared Error (MSE) penalizing deviations between network predictions and observed clinical measurements.
2. **Physics/Biological Loss ($\mathcal{L}_{\text{physics}}$):** Leverages PyTorch automatic differentiation (`torch.autograd`) to evaluate the exact structural residual of the ODE system at dense collation time steps.

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{data}} + \lambda \cdot \mathcal{L}_{\text{physics}}$$

Where $\lambda$ enforces strict alignment with human homeostatic limits.

## 🛠️ Repository Architecture

```text
├── main.py                 # Main script: Data generation, PINN class, and training loop
├── pinn_reconstruction.png # Saved optimization trajectory graph
└── README.md               # Project documentation and engineering diagnostics
```

---

## 📊 Performance & Parameter Discovery Analytics

During optimization, the unknown biological parameter $p_3$ is initialized far from its true value to evaluate the inversion performance of the machine learning framework:

- **True Ground-Truth Parameter ($p_3$):** `0.01000`
- **PINN Back-Calculated Discovery Value ($p_3$):** `0.00998`
- **Estimation Metric Accuracy:** `~99.8%` verification

### Trajectory Reconstruction Visual

The PINN filters out the Gaussian device noise to reveal the true underlying continuous physiological clearance path:

![PINN Trajectory Reconstruction](pinn_reconstruction.png)

---

## ⚡ Local Setup & Execution

### Prerequisites

Ensure your environment uses **Python 3.10+** and install the required scientific computing packages:

```bash
pip install torch numpy matplotlib
```
---

### Option A: Run from Terminal

Clone or download the repository and execute:

```bash
python main.py
```

---

### Option B: Run via Python IDLE

1. Launch Python IDLE.
2. Select **File → Open**.
3. Open `main.py`.
4. Press **F5** to run the script.

When executed locally, the script prints step-by-step training updates to the shell, outputs the final back-calculated parameters, and launches a native, interactive plot window.

## 🎯 Relevancy to Deep Scientific Machine Learning

This independent project demonstrates:

- Formulating custom structural losses utilizing neural network automatic differentiation.
- Deploying deep learning pipelines capable of executing inverse modelling for translational healthcare applications.
- Interdisciplinary software implementation connecting advanced mathematical systems theory to real-world neural architectures.

## 📄 License

Distributed under the MIT License. See `LICENSE` for more information.
