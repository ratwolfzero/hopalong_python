import numpy as np
import matplotlib.pyplot as plt
from numba import njit
from math import copysign, sqrt, fabs

# Input validation
def validate_input(prompt, input_type=float, check_positive_non_zero=False, min_value=None):
    while True:
        user_input = input(prompt)
        try:
            value = float(user_input)
            if input_type == int and not value.is_integer():
                print('Invalid input. Please enter an integer.')
                continue
            value = int(value) if input_type == int else value
            if check_positive_non_zero and value <= 0:
                print('Invalid input. Please enter a positive non-zero number.')
                continue
            if min_value is not None and value < min_value:
                print(f'Invalid input. The value should be at least {min_value}.')
                continue
            return value
        except ValueError:
            print(f'Invalid input. Please enter a valid {input_type.__name__} value.')

# Get attractor parameters
def get_attractor_parameters():
    a = validate_input('Enter a float value for "a": ', float)
    b = validate_input('Enter a float value for "b": ', float)
    while True:
        c = validate_input('Enter a float value for "c": ', float)
        if (a == 0 and b == 0 and c == 0) or (a == 0 and c == 0):
            print('Invalid combination of parameters. Please re-enter.')
        else:
            break
    n = validate_input('Enter a positive integer value > 1000 for "n": ', int, check_positive_non_zero=True, min_value=1000)
    return {'a': a, 'b': b, 'c': c, 'n': n}

# Compute trajectory extents
@njit
def compute_trajectory_extents(a, b, c, n):
    x, y = np.float64(0.0), np.float64(0.0)
    min_x, max_x, min_y, max_y = np.inf, -np.inf, np.inf, -np.inf
    for _ in range(n):
        min_x, max_x = min(min_x, x), max(max_x, x)
        min_y, max_y = min(min_y, y), max(max_y, y)
        xx = y - copysign(1.0, x) * sqrt(fabs(b * x - c))
        yy = a - x
        x, y = xx, yy
    return min_x, max_x, min_y, max_y

# Compute trajectory image
@njit
def compute_trajectory_image(a, b, c, n, extents, image_size):
    image = np.zeros(image_size, dtype=np.uint64)
    min_x, max_x, min_y, max_y = extents
    scale_x = (image_size[1] - 1) / (max_x - min_x)
    scale_y = (image_size[0] - 1) / (max_y - min_y)
    x, y = np.float64(0.0), np.float64(0.0)

    for _ in range(n):
        px = round((x - min_x) * scale_x)
        py = round((y - min_y) * scale_y)
        if 0 <= px < image_size[1] and 0 <= py < image_size[0]:
            image[py, px] += 1
        xx = y - copysign(1.0, x) * sqrt(fabs(b * x - c))
        yy = a - x
        x, y = xx, yy

    return image

# Compute correlation integral with optimized approach
@njit
def compute_correlation_integral(image, r):
    count = 0
    total_pairs = 0
    kernel_size = 2 * r + 1
    for x in range(image.shape[0]):
        for y in range(image.shape[1]):
            if image[x, y] > 0:
                count += image[x, y] * np.sum(image[max(0, x - r):min(x + r + 1, image.shape[0]),
                                                    max(0, y - r):min(y + r + 1, image.shape[1])])
                total_pairs += kernel_size ** 2
    return count / total_pairs if total_pairs > 0 else 0

# Estimate correlation dimension
def estimate_correlation_dimension(image, r_values):
    correlations = []
    for r in r_values:
        correlation = compute_correlation_integral(image, int(r))
        correlations.append(correlation)

    # Fit a power law to the correlation integral
    log_r = np.log(r_values)
    log_correlations = np.log(correlations)
    slope, intercept = np.polyfit(log_r, log_correlations, 1)

    return slope, correlations  # Return both dimension and correlations

# Plot results
def plot_correlation_integral(r_values, correlations):
    plt.loglog(r_values, correlations, marker='o')
    plt.xlabel('r')
    plt.ylabel('C(r)')
    plt.title('Correlation Integral')
    plt.grid(True, which='both')
    plt.show()

# Main function
def main():
    params = get_attractor_parameters()
    extents = compute_trajectory_extents(params['a'], params['b'], params['c'], params['n'])
    image_size = (1000, 1000)  # Increased resolution
    r_values = np.logspace(0, 2, 20)  # Adjusted range for r_values

    image = compute_trajectory_image(params['a'], params['b'], params['c'], params['n'], extents, image_size)

    # Diagnostic plot for trajectory
    plt.imshow(image, extent=(extents[0], extents[1], extents[2], extents[3]), origin='lower', cmap='hot')
    plt.title('Trajectory Heatmap')
    plt.colorbar()
    plt.show()

    correlation_dimension, correlations = estimate_correlation_dimension(image, r_values)
    print(f"Correlation Dimension: {correlation_dimension:.4f}")

    plot_correlation_integral(r_values, correlations)

if __name__ == '__main__':
    main()
