import matplotlib.pyplot as plt
import numpy as np
from numba import jit

# Use TkAgg backend for matplotlib
import matplotlib
matplotlib.use('TkAgg')


@jit(nopython=True)
def hopalong_compute(num, a, b, c):
   # Computes the points for the Hopalong attractor. 
    points = np.empty((num, 2), dtype=np.float32)
    x, y = 0.0, 0.0

    for i in range(num):

        points[i] = x, y
        xx, yy = y - np.sign(x) * np.sqrt(abs(b * x - c)), a - x
        x, y = xx, yy

    return points


def hopalong_plot(points, a, b, c, num, image_size):
    # Plots the points of the Hopalong attractor.
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
    plt.imshow(img, cmap='inferno', extent=[min_x, max_x, min_y, max_y])
    plt.title(
        f"Hopalong Attractor@ratwolf2024\nParams: a={a}, b={b}, c={c}, num={num:_}")
    plt.show()


def hopalong(num, a, b, c, image_size):
    # Computes and plots the Hopalong attractor.

    # This function is split into compute and plot parts because the @jit
    # decorator does not support the plotting operations.

    # The points variables in the functions "hopalong_compute" and "hopalong" are different local variables.
    # They exist in different scopes and have their own lifetimes!
    points = hopalong_compute(num, a, b, c).astype(np.float32)
    hopalong_plot(points, a, b, c, num, image_size)


def get_validated_input(prompt, input_type=float, check_non_zero=False, check_num=False):
    # Prompts the user for input and validates it.
    while True:
        user_input = input(prompt)
        try:
            value = input_type(user_input)
            if check_non_zero and value == 0:
                print("Invalid input. The value cannot be zero.")
                continue
            if check_num and value < 1_000_000:
                print(
                    "Inappropriate input. The value for num should be at least 1,000,000.")
                continue
            return value
        except ValueError:
            print(f"Invalid input. Please enter a valid {
                  input_type.__name__} value.")


def main():
   # Main function to run the Hopalong attractor generation and plotting.
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
