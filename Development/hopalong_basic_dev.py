# Use TkAgg backend
import matplotlib
matplotlib.use('TkAgg')

import matplotlib.pyplot as plt
import numpy as np
from numba import njit, prange


@njit
def calculate_hopalong_attractor_points(a, b, c, num):
    # generate hopalong points array of given shape
    """
    Remark: The "parallel=true" option for @njit respectively prange cannot be used here due to the cross-iteration dependency
    points[i+1] cannot be calculated without first computing points[i]
    """
    points = np.zeros((num, 2), dtype=np.float32)
    x = y = 0.0

    for i in range(num):

        points[i] = x, y
        xx, yy = y - np.sign(x) * np.sqrt(abs(b * x - c)), a - x
        x, y = xx, yy

    return points


@njit(parallel=True)
def map_attractor_points_to_image_pixels(points, image_size):
    # Convert hopalong attractor points to image pixel locations
    img_width, img_height = image_size

    min_x, max_x = np.min(points[:, 0]), np.max(points[:, 0])
    min_y, max_y = np.min(points[:, 1]), np.max(points[:, 1])

    px = ((points[:, 0] - min_x) / (max_x - min_x) * (img_width - 1)).astype(np.int32) 
    py = ((points[:, 1] - min_y) / (max_y - min_y) * (img_height - 1)).astype(np.int32)

    return px, py, min_x, max_x, min_y, max_y


@njit(parallel=True)
def generate_image_and_pixel_counts(img, px, py):
    """
    Populate the image array with hit counts for each pixel
    this variant enables the use of parallel=true & prange!
    """
    for i in prange(len(px)):
        img[px[i], py[i]] += 1

    return img


def prepare_plot(points, a, b, c, num, image_size):
    # Process the attractor points and prepare data for plotting
    px, py, min_x, max_x, min_y, max_y = map_attractor_points_to_image_pixels(points, image_size)
    img = generate_image_and_pixel_counts(np.zeros(image_size, dtype=np.int32), px, py)

    extents = [min_x, max_x, min_y, max_y]
    params = {'a': a, 'b': b, 'c': c, 'num': num}

    return img, extents, params


def plot_hopalong_attractor(img, extents, params, color_map):
    # plot the hopalong attractor image
    plt.figure(figsize=(8, 8))
    plt.imshow(img, origin="lower", cmap=color_map, extent=extents)
    plt.title(
        "Hopalong Attractor@ratwolf@2024\nParams: a={a}, b={b}, c={c}, num={num:_}".format(**params))
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


def get_user_inputs():
    #Prompt for user input
    a = get_validated_input('Enter a non-zero float value for "a": ', float, check_non_zero=True)
    b = get_validated_input('Enter a float value for "b": ', float)
    c = get_validated_input('Enter a float value for "c": ', float)
    num = get_validated_input('Enter an integer value for "num": ', int, check_non_zero=True)

    return a, b, c, num


def coordinate_process_execution(a, b, c, num, image_size, color_map):
    points = calculate_hopalong_attractor_points(a, b, c, num)
    img, extents, params = prepare_plot(points, a, b, c, num, image_size)
    plot_hopalong_attractor(img, extents, params, color_map)


def main():
    # define image size and colormap
    image_size = 1000, 1000
    color_map = 'inferno'

    #Prompt for user inputs
    a, b, c, num = get_user_inputs()

    # coordinate and trigger the program execution
    coordinate_process_execution(a, b, c, num, image_size, color_map)


if __name__ == "__main__":
    main()

