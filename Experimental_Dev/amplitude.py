import matplotlib.pyplot as plt
import numpy as np
from numba import njit


@njit
def hopalong_attractor(a, b, c, iterations=None):
    x_vals = np.zeros(iterations)
    y_vals = np.zeros(iterations)

    x, y = 0.0, 0.0  # Initial values

    for i in range(iterations):
        x, y = y - np.sign(x) * np.sqrt(abs(b * x - c))  , a - x
        
        x_vals[i] = x
        y_vals[i] = y

    return x_vals, y_vals
    

def plot_hopalong_combined(x_vals, y_vals):
    iterations = range(len(x_vals))
    amplitude = np.sqrt(np.diff(x_vals, prepend=x_vals[0])**2 + np.diff(y_vals, prepend=y_vals[0])**2)

    plt.figure(figsize=(12, 10))

    plt.subplot(2, 2, 1, aspect='equal')
    plt.scatter(x_vals, y_vals, s=0.5, c='blue', alpha=0.7)
    plt.title("Hopalong Attractor")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.grid(True)

    plt.subplot(2, 1, 2)
    plt.plot(iterations, amplitude, label="Amplitude (Point Distance)", color="blue", alpha=0.7)
    plt.title("Amplitude of Hopalong Attractor Over Events")
    plt.xlabel("Iteration Index")
    plt.ylabel("Amplitude (Distance)")
    plt.xscale('log')
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.show()

def main():
	
    a, b, c = 0.7, 0.4, 0
    iterations = 10000000

    x_vals, y_vals = hopalong_attractor(a, b, c, iterations)

    plot_hopalong_combined(x_vals, y_vals)


if __name__ == "__main__":
    main()



