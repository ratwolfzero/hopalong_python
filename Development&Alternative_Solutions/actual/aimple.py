import matplotlib.pyplot as plt
import numpy as np
from numba import njit
from math import sqrt, copysign
import time


# Start the time measurement
start_time = time.process_time()

# Parameters for the Hopalong attractor
a = -2
b = -0.33
c = 0.01
iterations = 200000000
width, height = 1000, 1000  # Resolution of the pixel grid

# Initial values
x0, y0 = 0.0, 0.0

@njit
def calculate_extents(x0, y0, a, b, c, iterations):
    x, y = x0, y0
    x_min, x_max, y_min, y_max = x, x, y, y
    for _ in range(iterations):
        x_next = y - copysign(1,x) * sqrt(abs(b * x - c))
        y = a - x
        x = x_next
        x_min, x_max = min(x_min, x), max(x_max, x)
        y_min, y_max = min(y_min, y), max(y_max, y)
    return x_min, x_max, y_min, y_max

@njit
def create_density_map(x0, y0, a, b, c, iterations, x_min, x_max, y_min, y_max, width, height):
    x, y = x0, y0
    pixel_grid = np.zeros((height, width), dtype=np.int32)

    for _ in range(iterations):
        x_next = y - copysign(1,x) * sqrt(abs(b * x - c))
        y = a - x
        x = x_next

        # Map the continuous x, y values to integer pixel indices
        px = int((x - x_min) / (x_max - x_min) * (width - 1))
        py = int((y - y_min) / (y_max - y_min) * (height - 1))

        # Increment the pixel count for density mapping
        if 0 <= px < width and 0 <= py < height:
            pixel_grid[py, px] += 1

    return pixel_grid

# First pass to find extents
x_min, x_max, y_min, y_max = calculate_extents(x0, y0, a, b, c, iterations)

# Second pass to create density map
density_map = create_density_map(x0, y0, a, b, c, iterations, x_min, x_max, y_min, y_max, width, height)

# Plot the result using a colormap
plt.figure(figsize=(8, 8))
plt.imshow(density_map, cmap='hot', origin='lower',
           extent=(x_min, x_max, y_min, y_max))
           
           
plt.colorbar(label='Pixel Density')
plt.title(f'Hopalong Attractor Density with a={a}, b={b}, c={c}')
plt.xlabel('x')
plt.ylabel('y')


#ax.axis('equal')
plt.tight_layout()
#plt.show()

plt.pause(1)
plt.close()

# End the time measurement
end_time = time.process_time()

# Calculate the CPU user and system time
cpu_sys_time_used = end_time - start_time

print(f'CPU User&System time: {cpu_sys_time_used:.2f} seconds')

