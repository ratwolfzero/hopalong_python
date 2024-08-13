import numpy as np
import matplotlib.pyplot as plt


# Parameters for the Hopalong attractor
a = 1
b = 2
c = 3

# Number of iterations
iterations = 100
# Initialize arrays to hold the points
x = np.zeros(iterations)
y = np.zeros(iterations)

# Initial values
x[0] = 0
y[0] = 0

def custom_sign(x):
    """
    for floating point according IEEE 754 (e.g. like implemented in Rust)
    1.0 if the number is positive, +0.0 or INFINITY
    -1.0 if the number is negative, -0.0 or NEG_INFINITY
    NaN if the number is NaN
    """
    if np.isnan(x):
        return np.nan
    elif x > 0 or x == 0.0:
        return 1.0
    else:
        return -1.0

# Iterate the Hopalong attractor equations
for n in range(iterations - 1):
	# x[n + 1] = y[n] - np.sign(x[n]) * np.sqrt(abs(b * x[n] - c))
    x[n + 1] = y[n] - custom_sign(x[n]) * np.sqrt(abs(b * x[n] - c))
    y[n + 1] = a - x[n]
	
	
"""
#Iterate the Hopalong attractor equations using custom signum.
#With this user-defined Signum function, some borderline cases regarding the input parameters a, b and c will behave differntly
for n in range(iterations - 1):
	x[n + 1] = y[n] - custom_sign(x[n]) * np.sqrt(abs(b * x[n] - c))
	y[n + 1] = a - x[n]
"""
	
print(x)
print(y)

# Plot the results
"""
plt.figure(figsize=(8, 8))
plt.plot(x, y, 'o', c='red', markersize=0.2)
plt.title(f'Hopalong Attractor with a={a}, b={b}, c={c}')
plt.xlabel('x')
plt.ylabel('y')
plt.show()
"""


