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
# PINN Model
# ==============================================================================
class GlucosePINN(nn.Module):
    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(1, 32),
            nn.Tanh(),
            nn.Linear(32, 32),
            nn.Tanh(),
            nn.Linear(32, 2)
        )

    def forward(self, t):
        return self.net(t)


model = GlucosePINN()

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# ==============================================================================
# Physics Collocation Points
# ==============================================================================
t_physics = torch.linspace(
    0,
    120,
    100,
    requires_grad=True
).view(-1, 1)

epochs = 2000

print("Training PINN...")

# ==============================================================================
# Training Loop
# ==============================================================================
for epoch in range(epochs):

    optimizer.zero_grad()

    # -----------------------------
    # Data Loss
    # -----------------------------
    pred_data = model(t_data)

    G_pred = pred_data[:, 0:1]

    loss_data = nn.MSELoss()(G_pred, G_data)

    # -----------------------------
    # Physics Loss
    # -----------------------------
    pred_phys = model(t_physics)

    G_phys = pred_phys[:, 0:1]
    X_phys = pred_phys[:, 1:2]

    dG_dt = torch.autograd.grad(
        G_phys,
        t_physics,
        grad_outputs=torch.ones_like(G_phys),
        create_graph=True
    )[0]

    residual = dG_dt - (
        -p1 * (G_phys - G_b)
        - X_phys * G_phys
    )

    loss_physics = torch.mean(residual ** 2)

    # -----------------------------
    # Total Loss
    # -----------------------------
    loss = loss_data + 1.5 * loss_physics

    loss.backward()
    optimizer.step()

    if epoch % 400 == 0:
        print(
            f"Epoch {epoch:4d} | "
            f"Total Loss = {loss.item():.4f}"
        )

# ==============================================================================
# Prediction
# ==============================================================================
print("Training complete.")

with torch.no_grad():

    t_test = torch.linspace(0, 120, 200).view(-1, 1)

    prediction = model(t_test)

    G_reconstructed = prediction[:, 0].numpy()

# ==============================================================================
# Plot
# ==============================================================================
plt.figure(figsize=(10, 6))

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
    linewidth=3,
    label="PINN Reconstructed Glucose Trajectory"
)

plt.axhline(
    G_b,
    linestyle="--",
    color="gray",
    label="Basal Glucose Level"
)

plt.title(
    "Physics-Informed Neural Network Glucose Trajectory Reconstruction"
)

plt.xlabel("Time (minutes)")
plt.ylabel("Glucose Concentration (mg/dL)")

plt.grid(True, linestyle=":")
plt.legend()

plt.tight_layout()

plt.savefig(
    "pinn_reconstruction.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()
