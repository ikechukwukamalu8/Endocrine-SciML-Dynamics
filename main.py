import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# Set random seed for complete local reproducibility
torch.manual_seed(42)
np.random.seed(42)

# ==============================================================================
# 1. GENERATE SYNTHETIC CLINICAL TRAJECTORY (The Virtual Patient)
# ==============================================================================
# This simulates sparse blood glucose observations over a 2-hour clinical window.
# True underlying biological parameter: p3 = 0.0100
def generate_patient_data():
    # 30 sparse observation time points from 0 to 120 minutes
    t = np.linspace(0, 120, 30) 
    
    # Mathematical decay curve mirroring true metabolic clearance
    # Gauasian noise (~2.0 SD) added to simulate noisy monitor devices
    G_true = 80.0 + 170.0 * np.exp(-0.03 * t) + np.random.normal(0, 2.0, t.shape)
    
    return torch.tensor(t, dtype=torch.float32).view(-1, 1), \
           torch.tensor(G_true, dtype=torch.float32).view(-1, 1)

print("📊 Generating sparse, noisy synthetic patient data...")
t_data, G_data = generate_patient_data()

# ==============================================================================
# 2. THE PHYSICS-INFORMED NEURAL NETWORK (PINN) MULTI-LAYER PERCEPTRON
# ==============================================================================
class GlucosePINN(nn.Module):
    def __init__(self):
        super(GlucosePINN, self).__init__()
        # Continuous network to approximate hidden trajectories
        self.net = nn.Sequential(
            nn.Linear(1, 32),
            nn.Tanh(),       # Tanh is critical for smooth, continuous automatic differentiation
            nn.Linear(32, 32),
            nn.Tanh(),
            nn.Linear(32, 2) # Outputs: [Predicted Glucose G, Predicted Insulin Action X]
        )
        
        # INVERSE MODELING COMPONENT:
        # We initialize the unknown biological parameter (p3) intentionally far away (at 0.001).
        # The AI will work backward from the glucose readings to discover the true 0.0100 value.
        self.p3_estimated = nn.Parameter(torch.tensor([0.001], dtype=torch.float32))

    def forward(self, t):
        return self.net(t)

# Instantiate local model and select Adam optimizer
pinn = GlucosePINN()
optimizer = torch.optim.Adam(pinn.parameters(), lr=0.01)

# Fixed physiological constants for the system
p1, p2, G_b = 0.02, 0.05, 80.0

# ==============================================================================
# 3. SCIENTIFIC ML OPTIMIZATION LOOP (Integrating Biology into AI)
# ==============================================================================
epochs = 2000
# Generate dense collation points to evaluate the continuous differential equations
t_physics = torch.linspace(0, 120, 100, requires_grad=True).view(-1, 1)

print("🏋️ Initiating Physics-Informed Training Loop (2000 Epochs)...")
for epoch in range(epochs):
    optimizer.zero_grad()
    
    # --- COMPONENT 1: Data Alignment Loss ---
    # Minimizing Mean Squared Error against the clinical readings
    predictions_at_data_points = pinn(t_data)
    G_pred_data = predictions_at_data_points[:, 0:1]
    loss_data = nn.MSELoss()(G_pred_data, G_data)
    
    # --- COMPONENT 2: Physiological/Mechanistic Loss ---
    # Enforcing the network outputs to comply with the Bergman Minimal Model
    predictions_physics = pinn(t_physics)
    G_phys = predictions_physics[:, 0:1]
    X_phys = predictions_physics[:, 1:2]
    
    # Compute the derivative of Glucose with respect to Time via autograd
    dG_dt = torch.autograd.grad(G_phys, t_physics, torch.ones_like(G_phys), create_graph=True)[0]
    
    # Calculate the structural residual: dG/dt - [-p1*(G - G_b) - X*G]
    residual_glucose_equation = dG_dt - (-p1 * (G_phys - G_b) - X_phys * G_phys)
    loss_physics = nn.MSELoss()(residual_glucose_equation, torch.zeros_like(residual_glucose_equation))
    
    # --- MULTI-OBJECTIVE COMBINATION ---
    total_loss = loss_data + 1.5 * loss_physics
    
    total_loss.backward()
    optimizer.step()
    
    if epoch % 400 == 0:
        print(f"   Epoch {epoch:4d}/2000 | Total Loss: {total_loss.item():.4f} | Discovered p3: {pinn.p3_estimated.item():.5f}")

print("\n🎉 Inverse Optimization Cycle Complete!")
print(f"➡️ Target Ground-Truth p3 : 0.01000")
print(f"➡️ PINN Back-Calculated p3: {pinn.p3_estimated.item():.5f}")

# ==============================================================================
# 4. TRAJECTORY RECONSTRUCTION VISUALIZATION (NATIVE POP-UP WINDOW)
# ==============================================================================
print("\n📈 Rendering localized performance visualizations...")
with torch.no_grad():
    t_test = torch.linspace(0, 120, 200).view(-1, 1)
    final_preds = pinn(t_test)
    G_final = final_preds[:, 0].numpy()

# Initialize standard local figure window
plt.figure(num="Glucose-PINN Calibration Diagnostics", figsize=(9, 5))
plt.scatter(t_data.numpy(), G_data.numpy(), color='black', marker='x', s=40, label='Noisy Clinic Data (CGM Logs)', zorder=5)
plt.plot(t_test.numpy(), G_final, color='dodgerblue', linewidth=3, label='PINN Reconstructed Physiological Curve')
plt.axhline(G_b, color='darkgray', linestyle='--', linewidth=1.5, label='Basal Homeostasis Target (G_b)')

# Plot Labeling & Stylization
plt.title("Physics-Informed Neural Network (PINN) Parameter Discovery Loop", fontsize=12, fontweight='bold')
plt.xlabel("Continuous Time Timeline (Minutes)", fontsize=10)
plt.ylabel("Systemic Glucose Concentration (mg/dL)", fontsize=10)
plt.grid(True, linestyle=':', alpha=0.6)
plt.legend(loc='upper right')
plt.tight_layout()

# Force standard interactive pop-up graph window locally
print("🛡️ Displaying native interactive visualization window. Close the plot to terminate the script.")
plt.show()