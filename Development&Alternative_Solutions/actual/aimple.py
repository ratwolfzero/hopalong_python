import matplotlib.pyplot as plt
import numpy as np
from numba import njit
from math import sqrt, fabs, copysign
import time


# Start the time measurement
start_time = time.process_time()

# Parameters for the Hopalong attractor
a = -2.0
b = -0.33
c = 0.01
iterations = 200000000
#width, height = 1000, 1000  # Resolution of the pixel grid
image_size= 1000,1000

# Initial values
#x0, y0 = 0.0, 0.0
x0 = np.float64(0.0)
y0 = np.float64(0.0)

"""
@njit
def calculate_extents(x0, y0, a, b, c, iterations):
    x, y = x0, y0
    #x_min, x_max, y_min, y_max = x, x, y, y
    x_min = np.inf  # ensure that the initial minimum is determined correctly
    x_max = -np.inf # ensure that the initial maximum is determined correctly
    y_min = np.inf
    y_max = -np.inf
    for _ in range(iterations):
        x_next = y - copysign(1,x) * sqrt(fabs(b * x - c))
        y = a - x
        x = x_next
        x_min, x_max = min(x_min, x), max(x_max, x)
        y_min, y_max = min(y_min, y), max(y_max, y)
    return x_min, x_max, y_min, y_max
"""

@njit #njit is an alias for nopython=True
def calculate_extents(x, y,a, b, c, iterations):
    # Dynamically compute and track the minimum and maximum extents of the trajectory over 'num' iterations.
    x = np.float64(0.0)
    y = np.float64(0.0)

    x_min = np.inf  # ensure that the initial minimum is determined correctly
    x_max = -np.inf # ensure that the initial maximum is determined correctly
    y_min = np.inf
    y_max = -np.inf

    for _ in range(iterations):
    # selective min/max update using direct comparisons avoiding min/max function
        if x < x_min:
            x_min = x
        if x > x_max:
            x_max = x
        if y < y_min:
            y_min = y
        if y > y_max:
            y_max = y
        # signum function respecting the behavior of floating point numbers according to IEEE 754 (signed zero)
        xx = y - copysign(1.0, x) * sqrt(fabs(b * x - c))
        yy = a-x
        x = xx
        y = yy
        
    return x_min, x_max, y_min, y_max

@njit
def create_density_map(x, y, a, b, c, iterations, x_min, x_max, y_min, y_max, image_size):
    #x, y = x0, y0
    x = np.float64(0.0)
    y = np.float64(0.0)
    pixel_grid = np.zeros((image_size), dtype=np.uint64)

    for _ in range(iterations):
        x_next = y - copysign(1,x) * sqrt(fabs(b * x - c))
        y = a - x
        x = x_next

        # Map the continuous x, y values to integer pixel indices
        #px = np.uint64((x - x_min) / (x_max - x_min) * (width - 1))
        #py = np.uint64((y - y_min) / (y_max - y_min) * (height - 1))
        
        px = np.uint64((x - x_min) / (x_max - x_min) * (image_size[0]))
        py = np.uint64((y - y_min) / (y_max - y_min) * (image_size[1]))

        # Increment the pixel count for density mapping
        #if 0 <= px < width and 0 <= py < height:
        pixel_grid[py, px] += 1

    return pixel_grid

# First pass to find extents
x_min, x_max, y_min, y_max = calculate_extents(x0, y0, a, b, c, iterations)

# Second pass to create density map
density_map = create_density_map(x0, y0, a, b, c, iterations, x_min, x_max, y_min, y_max, image_size)

# Plot the result using a colormap
plt.figure(figsize=(8, 8))
plt.imshow(density_map, cmap='hot', origin='lower',
           extent=(x_min, x_max, y_min, y_max))
           
           
plt.colorbar(label='Pixel Density',location='bottom')
plt.title(f'Hopalong Attractor Density with a={a}, b={b}, c={c}')
plt.xlabel('x')
plt.ylabel('y')


#plt.axis('equal')
plt.tight_layout()
plt.show()

#plt.pause(1)
#plt.close()

# End the time measurement
end_time = time.process_time()

# Calculate the CPU user and system time
cpu_sys_time_used = end_time - start_time

print(f'CPU User&System time: {cpu_sys_time_used:.2f} seconds')

