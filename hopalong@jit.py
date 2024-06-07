import matplotlib.pyplot as plt
import numpy as np
from numba import jit
import matplotlib
matplotlib.use('TkAgg')


@jit(nopython=True)
def hopalong_compute(num, a, b, c):

    points = np.empty((num, 2), dtype=np.float32)
    x, y = 0.0, 0.0

    for i in range(num):

        points[i] = x, y
        xx, yy = y - np.sign(x) * np.sqrt(abs(b * x - c)), a - x
        x, y = xx, yy

    return points


def hopalong_plot(u, v, a, b, c, num, image_size):

    min_x, max_x = np.min(u), np.max(u)
    min_y, max_y = np.min(v), np.max(v)

    img_width, img_height = image_size
    img = np.empty((img_height, img_width), dtype=np.int16)

    px = ((u - min_x) / (max_x - min_x) * (img_width - 1)).astype(np.int16)
    py = ((v - min_y) / (min_y - max_y) * (img_height - 1)).astype(np.int16)

    img[py, px] = 1

    plt.figure(figsize=(10, 10))
    plt.imshow(img, cmap='inferno')
    plt.title(f"Hopalong Attractor@ratwolf2024\nParams: a={
        a}, b={b}, c={c}, num={(f"{num:_}")}")
    plt.show()


def hopalong(num, a, b, c, image_size):
    points = hopalong_compute(num, a, b, c)
    hopalong_plot(points[:, 0], points[:, 1], a, b, c, num, image_size)


def get_validated_input(prompt, input_type=float, check_non_zero=False, check_num=False):
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

        if check_num and value < 1_000_000:
            print("inappropriate input. The value for num should be at least 1_000_000.")
        else:
            return value


def main():
    image_size = 10000, 10000
    a = get_validated_input(
        'Enter a non-zero float value for "a": ', float, check_non_zero=True)
    b = get_validated_input('Enter a float value for "b": ', float)
    c = get_validated_input('Enter a float value for "c": ', float)
    num = get_validated_input(
        'Enter an integer value for "num": ', int, check_num=True)

    hopalong(num, a, b, c, image_size)


if __name__ == "__main__":
    main()
