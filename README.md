# Physics-Informed Neural Networks (PINNs) for Glucose Trajectory Reconstruction in Endocrine Dynamics

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![Framework: PyTorch](https://img.shields.io/badge/Framework-PyTorch-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 📌 Project Overview

This repository demonstrates the use of **Physics-Informed Neural Networks (PINNs)** for reconstructing glucose dynamics from sparse and noisy clinical observations.

Unlike conventional neural networks that rely solely on observational data, PINNs incorporate known physiological principles directly into the learning process through differential equation constraints. This enables the model to produce biologically plausible trajectories while maintaining consistency with established endocrine dynamics.

The project serves as an example of **Scientific Machine Learning (SciML)**, combining mechanistic knowledge with deep learning to improve interpretability, robustness, and physiological realism.

---

## 🧬 Mathematical Framework

The glucose dynamics are constrained by a simplified physiological model:

$$
\frac{dG}{dt}
=============

## -p_1\bigl(G(t)-G_b\bigr)

X(t)G(t)
$$

where:

* (G(t)) = blood glucose concentration
* (X(t)) = latent insulin action state
* (G_b) = basal glucose concentration
* (p_1) = glucose clearance coefficient

The neural network learns continuous approximations of both (G(t)) and (X(t)) while enforcing consistency with the governing differential equation.

---

## ⚙️ Physics-Informed Loss Function

The optimization objective combines two complementary components:

### Data Loss

The model minimizes the discrepancy between predicted glucose values and observed clinical measurements:

$$
\mathcal{L}_{data}
==================

\text{MSE}
\left(
G_{pred},
G_{obs}
\right)
$$

### Physics Loss

Automatic differentiation is used to compute temporal derivatives and enforce compliance with the physiological ODE:

$$
\mathcal{L}_{physics}
=====================

\text{MSE}
\left(
\text{ODE Residual}
\right)
$$

### Total Loss

$$
\mathcal{L}_{total}
===================

\mathcal{L}*{data}
+
\lambda
\mathcal{L}*{physics}
$$

where (\lambda) controls the balance between data fidelity and physiological consistency.

---

## 🛠️ Repository Architecture

```text
├── main.py
├── pinn_reconstruction.png
└── README.md
```

* **main.py** – PINN implementation, synthetic data generation, training loop, and visualization.
* **pinn_reconstruction.png** – Example trajectory reconstruction output.
* **README.md** – Project documentation.

---

## 📊 Trajectory Reconstruction Performance

The Physics-Informed Neural Network reconstructs the underlying glucose trajectory from sparse and noisy observations while satisfying physiological constraints.

Key observations:

* Smooth reconstruction of glucose decay dynamics.
* Robustness to simulated sensor noise.
* ODE-constrained optimization improves physiological realism.
* Demonstrates integration of mechanistic modeling and deep learning.

### Example Output

![PINN Trajectory Reconstruction](pinn_reconstruction.png)

---

## ⚡ Installation

Install required packages:

```bash
pip install torch numpy matplotlib
```

---

## ▶️ Running the Project

Execute:

```bash
python main.py
```

The script will:

1. Generate synthetic noisy glucose observations.
2. Train a Physics-Informed Neural Network.
3. Reconstruct the continuous glucose trajectory.
4. Display the final visualization.

---

## 🎯 Scientific Machine Learning Concepts Demonstrated

This project demonstrates:

* Physics-Informed Neural Networks (PINNs)
* Scientific Machine Learning (SciML)
* Automatic Differentiation
* ODE-Constrained Learning
* Hybrid Mechanistic–Data-Driven Modeling
* Computational Endocrinology
* Trajectory Reconstruction
* Interpretable Machine Learning

---

## 🚀 Future Directions

Potential future extensions include:

* Full Bergman Minimal Model implementation
* Hidden physiological parameter discovery
* Inverse modeling of insulin sensitivity parameters
* Bayesian PINNs for uncertainty quantification
* Personalized digital twins
* Real Continuous Glucose Monitoring (CGM) datasets
* Neural ODE and DeepONet comparisons

---

## 📄 License

Distributed under the MIT License. See `LICENSE` for more information.
