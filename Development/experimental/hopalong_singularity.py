import matplotlib.pyplot as plt
import numpy as np
from numba import njit
from math import copysign


# Parameters for the Hopalong attractor
a = 1
b = 1
c = 1

# Number of iterations
iterations = 10000
# Initialize arrays to hold the points
x = np.zeros(iterations)
y = np.zeros(iterations)

# Initial values
x[0] = 0.0
y[0] = 0.0

def custom_sign(x):
    
    if np.isnan(x):
        return np.nan
    elif x > 0 or x == 0.0:
        return 1.0
    else:
        return -1.0

# Iterate the Hopalong attractor equations
for n in range(iterations - 1):
    #x[n + 1] = y[n] - np.sign(x[n]) * np.sqrt(abs(b * x[n] - c))
    #x[n + 1] = y[n] - custom_sign(x[n]) * np.sqrt(abs(b * x[n] - c))
    x[n + 1] = y[n] - copysign(1.0, x[n]) * np.sqrt(abs(b * x[n] - c))
    y[n + 1] = a - x[n]
	
	
#print(x)
#print(y)

# Plot the results

plt.figure(figsize=(8, 8))
plt.plot(x, y, 'o', c='red', markersize=0.2)
plt.title(f'Hopalong Attractor with a={a}, b={b}, c={c}')
plt.xlabel('x')
plt.ylabel('y')
plt.show()



