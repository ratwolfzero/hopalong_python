import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

import numpy as np
from numba import njit, prange
from math import copysign, sqrt, fabs


def get_validated_input(prompt, input_type=float, check_non_zero=False, check_positive=False):
    """Prompt for and return user input validated by type and positive/non-zero checks."""
    while True:
        user_input = input(prompt)
        try:
            value = input_type(user_input)
            if check_non_zero and value == 0:
                print("Invalid input. The value cannot be zero.")
                continue
            if check_positive and value <= 0:
                print("Invalid input. The value must be a positive number.")
                continue
            return value
        except ValueError:
            print(f"Invalid input. Please enter a valid {input_type.__name__} value.")


def get_attractor_parameters():
    """Prompt user to input parameters for the Hopalong Attractor."""
    a = get_validated_input('Enter a float value for "a": ', float)
    b = get_validated_input('Enter a float value for "b": ', float)
    c = get_validated_input('Enter a float value for "c": ', float)
    num = get_validated_input('Enter a positive integer value for "num": ', int, check_non_zero=True, check_positive=True)
    params = {'a': a, 'b': b, 'c': c, 'num': num}
    return a, b, c, num, params


@njit
def compute_full_trajectory_extents(a, b, c, num):
    """Compute and return the minimum and maximum x and y values for the full attractor trajectory."""
    x = y = np.float64(0)
    min_x = min_y = np.inf
    max_x = max_y = -np.inf
    for i in range(num):
        min_x = min(min_x, x)
        max_x = max(max_x, x)
        min_y = min(min_y, y)
        max_y = max(max_y, y)
        xx = y - copysign(1.0, x) * sqrt(fabs(b * x - c))
        yy = a - x
        x, y = xx, yy
    extents = [min_x, max_x, min_y, max_y]
    return extents


@njit
def generate_chunk_sizes(num, chunk_size):
    """Yield sizes of chunks to process in each iteration until covering the entire range."""
    for i in range(0, num, chunk_size):
        current_chunk_size = min(chunk_size, num - i)
        yield current_chunk_size


@njit
def compute_trajectory_chunk(a, b, c, current_chunk_size, x0, y0):
    """Compute a chunk of the Hopalong trajectory."""
    points = np.zeros((current_chunk_size, 2), dtype=np.float64)
    x, y = x0, y0
    for i in range(current_chunk_size):
        points[i] = x, y
        xx = y - copysign(1.0, x) * sqrt(fabs(b * x - c))
        yy = a - x
        x, y = xx, yy
    return points, x, y


@njit(parallel=True)
def map_trajectory_chunk_to_image(image, points, extents):
    """Map trajectory chunk points to image pixel locations and populate the image accordingly."""
    min_x, max_x, min_y, max_y = extents
    img_width, img_height = image.shape[1], image.shape[0]
    px = ((points[:, 0] - min_x) / (max_x - min_x) * (img_width - 1)).astype(np.uint64)
    py = ((points[:, 1] - min_y) / (max_y - min_y) * (img_height - 1)).astype(np.uint64)
    for i in prange(len(px)):
        if 0 <= px[i] < img_width and 0 <= py[i] < img_height:
            image[py[i], px[i]] += 1


@njit(parallel=True)
def compute_full_trajectory_image(a, b, c, num, chunk_size, extents, image_size):
    """Calculate the full trajectory image from chunks."""
    img_width, img_height = image_size
    image = np.zeros((img_height, img_width), dtype=np.uint64)
    x0 = y0 = np.float64(0)
    for current_chunk_size in generate_chunk_sizes(num, chunk_size):
        points, x0, y0 = compute_trajectory_chunk(a, b, c, current_chunk_size, x0, y0)
        map_trajectory_chunk_to_image(image, points, extents)
    return image


def render_full_trajectory_image(image, extents, params, color_map):
    """Render the full trajectory image using matplotlib."""
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(1, 1, 1, aspect='auto')
    ax.imshow(image, origin="lower", cmap=color_map, extent=extents)
    ax.set_title("Hopalong Attractor@ratwolf@2024\nParams: a={a}, b={b}, c={c}, num={num:_}".format(**params))
    plt.show()


def main(image_size=(1000, 1000), color_map='hot', chunk_size=1048576):
    """Execute processes to generate and render the Hopalong Attractor."""
    try:
        a, b, c, num, params = get_attractor_parameters()
        extents = compute_full_trajectory_extents(a, b, c, num)
        image = compute_full_trajectory_image(a, b, c, num, chunk_size, extents, image_size)
        render_full_trajectory_image(image, extents, params, color_map)
    except Exception as e:
        print(f"An error occurred: {e}")


"""Main execution"""
if __name__ == "__main__":
    main()
