# Physics-Informed Neural Networks (PINNs) for Inverse Modelling and Parameter Discovery in Endocrine Dynamics

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![Framework: PyTorch](https://img.shields.io/badge/Framework-PyTorch-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 📌 Project Overview

This repository implements a **Physics-Informed Neural Network (PINN)** to solve the inverse problem of personalized parameter discovery in metabolic systems biology. Standard data-driven machine learning models are heavily prone to unphysiological anomalies when trained on sparse, noisy clinical datasets.

By embedding the non-linear **Bergman Minimal Model (ODEs)** directly into the neural network's backpropagation loss function, this framework uncovers hidden, patient-specific biological parameters—such as the insulin sensitivity/pancreatic supply metric (*p₃*)—from sparse clinical glucose trajectories.

---

## 🧬 Problem Statement & Mathematical Framework

In digital endocrinology, real-time monitors capture highly erratic and sparse time-series data. Traditional deep learning architectures cannot guarantee conservation laws or adherence to biological constraints.

This project bridges data-driven optimization with mechanistic dynamic simulation using a custom multi-objective loss landscape. The underlying physiological system state is governed by a modified set of differential equations tracking blood glucose concentration \(G(t)\) and remote insulin action activity \(X(t)\):

### Glucose Dynamics

\[
\frac{dG}{dt} = -p_1(G(t) - G_b) - X(t)G(t)
\]

### Insulin Action Dynamics

\[
\frac{dX}{dt} = -p_2X(t) + p_3 \cdot f(t)
\]

---

## 🎯 Multi-Objective Loss Design

The network minimizes a dual-component objective function during optimization:

### 1. Data Loss (\(\mathcal{L}_{data}\))

Mean Squared Error (MSE) penalizing deviations between network predictions and observed clinical measurements.

### 2. Physics Loss (\(\mathcal{L}_{physics}\))

Leverages PyTorch automatic differentiation (`torch.autograd`) to evaluate the structural residuals of the governing ODE system at dense collocation points.

### Total Objective Function

\[
\mathcal{L}_{total}
=
\mathcal{L}_{data}
+
\lambda \cdot \mathcal{L}_{physics}
\]

where \(\lambda\) controls the balance between observational fidelity and physiological consistency.

---

## 🛠️ Repository Architecture

```text
├── main.py                  # Data generation, PINN class, and training loop
├── pinn_reconstruction.png  # Reconstructed trajectory visualization
└── README.md                # Project documentation
```

---

## 📈 Trajectory Reconstruction

The PINN filters Gaussian measurement noise and reconstructs the latent physiological glucose clearance trajectory while simultaneously estimating unknown biological parameters.

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

Upon execution, the program:

- Displays training progress in the console.
- Estimates the unknown physiological parameters.
- Generates a reconstruction plot.
- Opens an interactive visualization window.

---

## 🔬 Scientific Machine Learning Relevance

This project demonstrates expertise in several areas of modern Scientific Machine Learning (SciML):

### Custom Physics-Constrained Loss Functions

Designing hybrid objectives that combine observational data with governing differential equations.

### Inverse Modelling in Medicine

Recovering latent biological parameters from sparse and noisy patient observations.

### Physics-Informed Deep Learning

Integrating mechanistic scientific knowledge directly into neural network optimization.

### Computational Systems Biology

Applying machine learning to dynamic physiological systems and endocrine regulation.

### Differentiable Scientific Computing

Leveraging automatic differentiation for solving constrained optimization and parameter estimation problems.

---

## 🚀 Potential Extensions

Future enhancements may include:

- Bayesian PINNs for uncertainty quantification.
- Personalized digital twin modelling.
- Neural ODE and latent ODE formulations.
- Multi-compartment glucose–insulin dynamics.
- Real clinical Continuous Glucose Monitoring (CGM) integration.
- Probabilistic parameter inference with variational methods.

---

## 📄 License

Distributed under the MIT License.

See the `LICENSE` file for additional information.
