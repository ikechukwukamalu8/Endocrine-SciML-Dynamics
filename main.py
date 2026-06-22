import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# ==============================================================================
# Reproducibility
# ==============================================================================
torch.manual_seed(42)
np.random.seed(42)

# ==============================================================================
# Generate Synthetic Glucose Data
# ==============================================================================
def generate_patient_data():
    t = np.linspace(0, 120, 30)
    # Clean exponential decay baseline + sensor noise
    G = (
        80.0
        + 170.0 * np.exp(-0.03 * t)
        + np.random.normal(0, 2.0, size=t.shape)
    )
    return (
        torch.tensor(t, dtype=torch.float32).view(-1, 1),
        torch.tensor(G, dtype=torch.float32).view(-1, 1),
    )

print("Generating synthetic glucose observations...")
t_data, G_data = generate_patient_data()

# ==============================================================================
# Physiological Constants
# ==============================================================================
p1 = 0.02
G_b = 80.0

# ==============================================================================
# PINN Model with Internal Input Normalization
# ==============================================================================
class GlucosePINN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 32),
            nn.Tanh(),
            nn.Linear(32, 32),
            nn.Tanh(),
            nn.Linear(32, 1)
        )
        
        # Smart Initialization: bias output to start near upper physiological range
        # to encourage the model to scale downwards smoothly
        with torch.no_grad():
            self.net[-1].bias.fill_(150.0)

    def forward(self, t):
        # Scale input down from [0, 120] to [0, 1] internally 
        # This prevents Tanh saturation pathologies completely
        t_scaled = t / 120.0
        return self.net(t_scaled)

model = GlucosePINN()

# Lower learning rate combined with Adam helps smooth gradient updates
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

# Physics Collocation Points
t_physics = torch.linspace(0, 120, 100, requires_grad=True).view(-1, 1)
epochs = 4000

print("Training PINN with Input Scaling...")

# ==============================================================================
# Training Loop
# ==============================================================================
for epoch in range(epochs):
    optimizer.zero_grad()

    # 1. Compute Data Loss
    G_pred = model(t_data)
    loss_data = nn.MSELoss()(G_pred, G_data)

    # 2. Compute Physics Loss
    G_phys = model(t_physics)
    dG_dt = torch.autograd.grad(
        G_phys,
        t_physics,
        grad_outputs=torch.ones_like(G_phys),
        create_graph=True
    )[0]
    
    residual = dG_dt + p1 * (G_phys - G_b)
    loss_physics = torch.mean(residual ** 2)

    # Clean Joint Optimization Loss Configuration
    loss = loss_data + loss_physics

    loss.backward()
    optimizer.step()

    if epoch % 500 == 0:
        print(
            f"Epoch {epoch:4d} | "
            f"Total Loss = {loss.item():.4f} | "
            f"Data Loss = {loss_data.item():.4f} | "
            f"Physics Loss = {loss_physics.item():.4f}"
        )

print("Training complete.")

# ==============================================================================
# Prediction & Inference
# ==============================================================================
with torch.no_grad():
    t_test = torch.linspace(0, 120, 200).view(-1, 1)
    prediction = model(t_test)
    G_reconstructed = prediction[:, 0].numpy()

# ==============================================================================
# Visualization Engine
# ==============================================================================
plt.figure(figsize=(11, 6))

plt.scatter(
    t_data.numpy(),
    G_data.numpy(),
    color="black",
    marker="x",
    s=60,
    label="Noisy Clinical Observations",
    zorder=5
)

plt.plot(
    t_test.numpy(),
    G_reconstructed,
    color="#1f77b4",
    linewidth=3,
    label="PINN Reconstructed Glucose Trajectory",
    zorder=4
)

plt.axhline(
    G_b,
    linestyle="--",
    color="gray",
    linewidth=1.5,
    label="Basal Glucose Level (G_b)"
)

plt.title("Physics-Informed Neural Network Glucose Trajectory Reconstruction", fontsize=12, pad=12)
plt.xlabel("Time (minutes)", fontsize=10)
plt.ylabel("Glucose Concentration (mg/dL)", fontsize=10)
plt.grid(True, linestyle=":", alpha=0.6)
plt.legend(loc="upper right", frameon=True)
plt.tight_layout()

# Save image file locally to your desktop workspace directory
plt.savefig("pinn_reconstruction.png", dpi=300, bbox_inches="tight")

# Generates pop-up interactive window inside local environments like IDLE
plt.show()
