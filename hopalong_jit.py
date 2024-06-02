import matplotlib.pyplot as plt
import numpy as np
from numba import jit
import matplotlib
matplotlib.use('TkAgg')

# Use TKAgg or Qt5Agg for MacOs to avoid crash of plot window during interaction

image_size = 8000, 8000


@jit(nopython=True)
def hopalong_compute(num, a, b, c):

    points = np.empty((num, 2), dtype=np.float32)
    x, y = 0.0, 0.0

    for i in range(num):

        points[i] = x, y
        xx, yy = y - np.sign(x) * np.sqrt(abs(b * x - c)), a - x
        x, y = xx, yy

    return points


def hopalong_plot(u, v, a, b, c, image_size):

    min_x, max_x = np.min(u), np.max(u)
    min_y, max_y = np.min(v), np.max(v)

    img_width, img_height = image_size
    img = np.empty((img_height, img_width), dtype=np.int16)

    px = ((u - min_x) / (max_x - min_x) * (img_width - 1)).astype(np.int16)
    py = ((v - min_y) / (min_y - max_y) * (img_height - 1)).astype(np.int16)

    img[py, px] = 1

    plt.figure(figsize=(8, 8))
    plt.imshow(img, cmap='inferno')
    plt.title(f"Hopalong Attractor\nParams: a={
        a}, b={b}, c={c}, num={(f"{num:_}")}")
    plt.show()


def hopalong(num, a, b, c, image_size):
    points = hopalong_compute(num, a, b, c)
    hopalong_plot(points[:, 0], points[:, 1], a, b, c, image_size)


print("Input the parameters a, b, c (e.g., -1.7 -0.3 0.7) and the number of iterations num (e.g., 1000000 or 1_000_000)")
# recommandation: use a maximum pf 300_000_000 iterations to avoid extensive swap of memory resulting in decrease of speed! (8 GByte RAM)


def get_validated_input(prompt, input_type=float, check_non_zero=False):
    while True:
        user_input = input(prompt)
        try:
            value = input_type(user_input)
        except ValueError:
            print(f"Invalid input. Please enter a valid {
                  input_type.__name__} value.")
            continue

        if check_non_zero and value == 0:
            print("Invalid input. The value cannot be zero.")
        else:
            return value


a = get_validated_input(

    'Enter a non-zero float value for "a": ', float, check_non_zero=True)
b = get_validated_input('Enter a float value for "b": ', float)
c = get_validated_input('Enter a float value for "c": ', float)
num = get_validated_input('Enter an integer value for "num": ', int)

hopalong(num, a, b, c, image_size)
