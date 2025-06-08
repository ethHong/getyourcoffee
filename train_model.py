import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from NN_model import *

# Load train data
df_long = pd.read_csv("baseline_train_data.csv")
df_onehot = pd.get_dummies(df_long["time_slot"], prefix="slot")
df_long = pd.concat([df_long, df_onehot], axis=1)
feature_cols = ["rating_inversed", "density_stretched", "proximity_curved"] + list(
    df_onehot.columns
)

X = df_long[feature_cols].values.astype(np.float32)
y = df_long["hypothetical_wait"].values.astype(np.float32).reshape(-1, 1)

X_tensor = torch.tensor(X)
y_tensor = torch.tensor(y)
model = GeneralizedSigmoid(input_dim=X_tensor.shape[1])
loss_fn = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

for epoch in range(1000):
    model.train()
    y_pred = model(X_tensor)
    loss = loss_fn(y_pred, y_tensor)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

# Plot results

X_tensor = torch.tensor(X).float()
with torch.no_grad():
    z_data = model.linear(X_tensor).squeeze().numpy()
    y_data = y

z_min = z_data.min()
z_max = z_data.max()


z_range = np.linspace(z_min, z_max, 300)
z_tensor = torch.tensor(z_range).unsqueeze(1).float()


with torch.no_grad():
    y_smooth = (
        model.L / (1 + torch.exp(-model.k * (z_tensor - model.x0))) + model.offset
    )


plt.figure(figsize=(8, 5))
plt.plot(
    z_range, y_smooth.numpy(), color="red", linewidth=2, label="Fitted Sigmoid Curve"
)
plt.scatter(z_data, y_data, alpha=0.6, label="Data Points", color="blue")
plt.xlabel("Linear Score (z = wᵀx + b)")
plt.ylabel("Predicted / Actual Wait Time")
plt.title("Generalized Sigmoid Fit with Data")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# Save the model:
date_now = pd.Timestamp.now().strftime("%Y-%m-%d_%H-%M-%S")
model_name = f"sigmoid_model_{date_now}.pt"
torch.save(model.state_dict(), f"model_update/{model_name}")
print(f"Model saved as {model_name}")
