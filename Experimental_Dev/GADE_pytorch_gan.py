import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from numba import njit



# --- Define Hopalong Attractor ---
def hopalong_attractor(a, b, c, n):
    x, y = 0.0, 0.0
    points = []
    for _ in range(n):
        x_new = y - np.sign(x) * np.sqrt(abs(b * x - c))
        y_new = a - x
        x, y = x_new, y_new
        points.append((x, y))
    return np.array(points, dtype=np.float32)


# Generate Hopalong attractor data
a, b, c = 0.6, 0.5, 0
n = 10000000
real_data = hopalong_attractor(a, b, c, n)

# --- GAN Components ---
# Generator
class Generator(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )
    
    def forward(self, z):
        return self.model(z)

# Discriminator
class Discriminator(nn.Module):
    def __init__(self, input_dim):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.model(x)

# --- Hyperparameters ---
latent_dim = 2  # Dimensionality of random input (z)
data_dim = 2    # Dimensionality of Hopalong attractor points
lr = 0.0002
epochs = 100000
batch_size = 64

# --- Model Initialization ---
generator = Generator(latent_dim, data_dim)
discriminator = Discriminator(data_dim)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
generator.to(device)
discriminator.to(device)

# --- Optimizers and Loss ---
optimizer_g = optim.Adam(generator.parameters(), lr=lr)
optimizer_d = optim.Adam(discriminator.parameters(), lr=lr)
loss_fn = nn.BCELoss()

# --- Data Preparation ---
real_data_tensor = torch.tensor(real_data, device=device)
real_labels = torch.ones(batch_size, 1, device=device)
fake_labels = torch.zeros(batch_size, 1, device=device)

# --- Training Loop ---
for epoch in range(epochs):
    # --- Train Discriminator ---
    discriminator.zero_grad()
    # Real data
    real_batch = real_data_tensor[torch.randint(0, len(real_data), (batch_size,))]
    real_pred = discriminator(real_batch)
    real_loss = loss_fn(real_pred, real_labels)
    
    # Fake data
    z = torch.randn(batch_size, latent_dim, device=device)
    fake_batch = generator(z)
    fake_pred = discriminator(fake_batch.detach())
    fake_loss = loss_fn(fake_pred, fake_labels)
    
    # Backpropagation
    d_loss = real_loss + fake_loss
    d_loss.backward()
    optimizer_d.step()

    # --- Train Generator ---
    generator.zero_grad()
    z = torch.randn(batch_size, latent_dim, device=device)
    fake_batch = generator(z)
    fake_pred = discriminator(fake_batch)
    g_loss = loss_fn(fake_pred, real_labels)
    
    # Backpropagation
    g_loss.backward()
    optimizer_g.step()

    # --- Logging ---
    if epoch % 100 == 0:
        print(f"Epoch {epoch}/{epochs} | D Loss: {d_loss.item():.4f} | G Loss: {g_loss.item():.4f}")

# --- Visualization ---
# Generate points from the trained generator
z = torch.randn(10000, latent_dim, device=device)
generated_data = generator(z).cpu().detach().numpy()

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.scatter(real_data[:, 0], real_data[:, 1], s=1, c='blue', label="Real Data")
plt.title("Real Hopalong Attractor")
plt.legend()

plt.subplot(1, 2, 2)
plt.scatter(generated_data[:, 0], generated_data[:, 1], s=1, c='red', label="Generated Data")
plt.title("Generated Data from GAN")
plt.legend()

plt.tight_layout()
plt.show()
