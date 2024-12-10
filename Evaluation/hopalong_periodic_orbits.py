import matplotlib.pyplot as plt
import numpy as np
from numba import njit

@njit
def custom_sign(x):
    if np.isnan(x):
        return np.nan
    elif x > 0 or x == 0.0:
        return 1.0
    else:
        return -1.0

@njit
def compute_hopalong(x, y, a, b, c, iterations):
    # Iterate the Hopalong attractor equations
    for n in range(iterations - 1):
        x[n + 1] = y[n] - custom_sign(x[n]) * np.sqrt(abs(b * x[n] - c))
        y[n + 1] = a - x[n]
    return x, y

# Plot the results
def plot_hopalong(x, y, a, b, c):
    plt.figure(figsize=(8, 8))
    plt.plot(x, y, 'o', c='red', markersize=4)
    plt.title(f'Hopalong Attractor with a={a}, b={b}, c={c}')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()

def main():
    # Define local parameters
    a = 5
    b = 5
    c = 0
    iterations = 1000000

    # Initialize arrays to hold the points
    x = np.zeros(iterations)
    y = np.zeros(iterations)

    # Initial values
    x[0] = 0.0
    y[0] = 0.0

    # Call the compute_hopalong function with arguments
    x_vals, y_vals = compute_hopalong(x, y, a, b, c, iterations)
    plot_hopalong(x_vals, y_vals, a, b, c)

main()

