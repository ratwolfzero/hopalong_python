import matplotlib.pyplot as plt
import numpy as np
import matplotlib
from numba import jit

matplotlib.use('TkAgg')

@jit(nopython=True)
def generate_points(num, a, b, c):
    u = np.zeros(num)
    v = np.zeros(num)
    x, y = 0, 0

    for i in range(num):
        u[i] = x
        v[i] = y
        xx = y - np.sign(x) * np.sqrt(abs(b * x - c))
        yy = a - x
        x = xx
        y = yy

    return u, v

def hopalong(num, a, b, c, image_size=(1001, 1001)):
    u, v = generate_points(num, a, b, c)

    min_x, max_x = np.min(u), np.max(u)
    min_y, max_y = np.min(v), np.max(v)

    img_width, img_height = image_size
    img = np.zeros((img_height, img_width), dtype=np.uint8)

    # Ensure proper scaling and flipping for y coordinates
    scale_x = (img_width - 1) / (max_x - min_x)
    scale_y = (img_height - 1) / (max_y - min_y)  # Corrected scaling factor

    px = ((u - min_x) * scale_x).astype(int)
    py = ((v - min_y) * scale_y).astype(int)

    # Ensure coordinates are within bounds
    px = np.clip(px, 0, img_width - 1)
    py = np.clip(py, 0, img_height - 1)

    # Use numpy advanced indexing to plot points
    np.add.at(img, (py, px), 1)

    # For better visualization, use logarithmic scaling
    img = np.log1p(img)

    plt.figure(figsize=(8, 8))
    plt.imshow(img, cmap='inferno', origin='lower')
    plt.title(f"Hopalong Attractor\nParams: a={a}, b={b}, c={c}, num={num}")
    plt.show()

# Input parameters
print("Input the parameters a, b, c (e.g., -1.7 -0.3 0.7) and the number of iterations num (e.g., 1000000 or 1_000_000)")
a = float(input('a? '))
b = float(input('b? '))
c = float(input('c? '))
num = int(input('num? '))
hopalong(num, a, b, c)