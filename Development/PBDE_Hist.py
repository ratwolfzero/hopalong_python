import numpy as np
import matplotlib.pyplot as plt

# Generate Gaussian-distributed data
##mean = [0, 0]  # Mean of the distribution (centered at origin)
cov = [[1, 0], [0, 1]]  # Covariance matrix (identity for independent variables)

# Draw 10,000 samples from the distribution
#x, y = np.random.multivariate_normal(mean, cov, 10000).T

# Generate non-uniformly distributed data (mixture of two Gaussians)
mean1 = [0, 0]  # Mean of the first Gaussian
cov1 = [[1, 0], [0, 1]]  # Covariance matrix for the first Gaussian
mean2 = [5, 5]  # Mean of the second Gaussian
cov2 = [[1, 0], [0, 1]]  # Covariance matrix for the second Gaussian

# Draw 10,000 samples from each distribution
x1, y1 = np.random.multivariate_normal(mean1, cov1, 50000).T
x2, y2 = np.random.multivariate_normal(mean2, cov2, 50000).T

# Combine the two sets of data
x = np.concatenate([x1, x2])
y = np.concatenate([y1, y2])

def PBDE(x, y, bins=(100, 100), x_range=None, y_range=None): #Pixel-Based Density Estimation
    # Determine ranges
    if x_range is None:
        x_range = (np.min(x), np.max(x))
    if y_range is None:
        y_range = (np.min(y), np.max(y))

    # Create grid
    x_edges = np.linspace(x_range[0], x_range[1], bins[0] + 1)
    y_edges = np.linspace(y_range[0], y_range[1], bins[1] + 1)

    # Map coordinates to pixels
    x_indices = np.floor((x - x_range[0]) / (x_range[1] - x_range[0]) * bins[0]).astype(int)
    y_indices = np.floor((y - y_range[0]) / (y_range[1] - y_range[0]) * bins[1]).astype(int)

    # Clip indices to stay within bounds
    x_indices = np.clip(x_indices, 0, bins[0] - 1)
    y_indices = np.clip(y_indices, 0, bins[1] - 1)

    # Aggregate density
    density = np.zeros((bins[0], bins[1]), dtype=int)
    np.add.at(density, (x_indices, y_indices), 1)

    # Calculate the total number of points
    total_points = len(x)

    # Normalize the density to match the behavior of np.histogram2d (density=True)
    bin_area = (x_range[1] - x_range[0]) / bins[0] * (y_range[1] - y_range[0]) / bins[1]
    density = density / (total_points * bin_area)

    
    return density, x_edges, y_edges
    

# Apply Pixel-Based Density Estimation (PBDE) with normalization
density_pbde, xedges, yedges = PBDE(x, y, bins=(100, 100))

# Apply 2D Histogram Approximation
density_hist, xedges_hist, yedges_hist = np.histogram2d(x, y, bins=(100, 100), range=[[xedges[0], xedges[-1]], [yedges[0], yedges[-1]]], density=True)

# Plot the results side by side for comparison

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Pixel-Based Density Estimation Plot (Normalized)
im1 = axes[0].imshow(
    density_pbde.T,
    origin='lower',
    extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
    cmap='hot',
    aspect='auto'
)
axes[0].set_title('Normalized Pixel-Based Density Estimation (PBDE)')
axes[0].set_xlabel('X')
axes[0].set_ylabel('Y')
plt.colorbar(im1, ax=axes[0], label='Density')  # Add colorbar for PBDE plot

# 2D Histogram Plot
im2 = axes[1].imshow(
    density_hist.T,
    origin='lower',
    extent=[xedges_hist[0], xedges_hist[-1], yedges_hist[0], yedges_hist[-1]],
    cmap='hot',
    aspect='auto'
)
axes[1].set_title('2D Histogram Approximation')
axes[1].set_xlabel('X')
axes[1].set_ylabel('Y')
plt.colorbar(im2, ax=axes[1], label='Density')  # Add colorbar for histogram plot

# Show the plot
plt.tight_layout()
plt.show()
