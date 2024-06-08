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


def hopalong_plot(points, a, b, c, num, image_size):
    min_x, max_x = np.min(points[:, 0]), np.max(points[:, 0])
    min_y, max_y = np.min(points[:, 1]), np.max(points[:, 1])

    img_width, img_height = image_size
    img = np.empty((img_height, img_width), dtype=np.int16)

    px = ((points[:, 0] - min_x) / (max_x - min_x)
          * (img_width - 1)).astype(np.int16)
    py = ((points[:, 1] - min_y) / (min_y - max_y)
          * (img_height - 1)).astype(np.int16)

    img[py, px] = 1

    plt.figure(figsize=(10, 10))
    plt.imshow(img, cmap='inferno')
    plt.title(
        f"Hopalong Attractor@ratwolf2024\nParams: a={a}, b={b}, c={c}, num={num:,}")
    plt.show()


def hopalong(num, a, b, c, image_size):
    points = hopalong_compute(num, a, b, c)   
    #Note: The points variable in the hopalong_compute function and the points variable in the hopalong function are different local variables. 
    #They exist in different scopes and have their own lifetimes!
    hopalong_plot(points, a, b, c, num, image_size)


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
