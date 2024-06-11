import matplotlib.pyplot as plt
import numpy as np
from numba import jit

# Use TkAgg backend for matplotlib
import matplotlib
matplotlib.use('TkAgg')


def hopalong(num, a, b, c, image_size):
    # Computes and plot the points for the Hopalong attractor.
    points = np.empty((num, 2), dtype=np.float32)
    x, y = 0.0, 0.0

    for i in range(num):

        points[i] = x, y
        xx, yy = y - np.sign(x) * np.sqrt(abs(b * x - c)), a - x
        x, y = xx, yy

    min_x, max_x = np.min(points[:, 0]), np.max(points[:, 0])
    min_y, max_y = np.min(points[:, 1]), np.max(points[:, 1])

    img_width, img_height = image_size
    img = np.empty((img_height, img_width), dtype=np.int16)

    px = ((points[:, 0] - min_x) / (max_x - min_x)
          * (img_width - 1)).astype(np.int16)
    py = ((points[:, 1] - min_y) / (max_y - min_y)
          * (img_height - 1)).astype(np.int16)

    img[py, px] = 1

    plt.figure(figsize=(8, 8))
    plt.imshow(img, origin="lower", cmap='inferno', extent=[min_x, max_x, min_y, max_y])
    plt.title(
        f"Hopalong Attractor@ratwolf2024\nParams: a={a}, b={b}, c={c}, num={num:_}")
    plt.show()


def get_validated_input(prompt, input_type=float, check_non_zero=False):
    # Prompts the user for input and validates it.
    while True:
        user_input = input(prompt)
        try:
            value = input_type(user_input)
            if check_non_zero and value == 0:
                print("Invalid input. The value cannot be zero.")
                continue
            return value
        except ValueError:
            print(f"Invalid input. Please enter a valid {
                  input_type.__name__} value.")


def main():
    # Main function to run the Hopalong attractor generation.
    image_size = 1000, 1000
    a = get_validated_input(
        'Enter a non-zero float value for "a": ', float, check_non_zero=True)
    b = get_validated_input('Enter a float value for "b": ', float)
    c = get_validated_input('Enter a float value for "c": ', float)
    num = get_validated_input('Enter an integer value for "num": ', int)

    hopalong(num, a, b, c, image_size)


if __name__ == "__main__":
    main()
