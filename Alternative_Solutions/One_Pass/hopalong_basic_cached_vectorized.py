"""Use TkAgg backend"""
import matplotlib; matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

import numpy as np
from numba import njit, prange
from math import copysign, sqrt, fabs
import time


def get_validated_input(prompt, input_type=float, check_non_zero=False, check_positive=False):
    #Prompt for and return user input validated by type and positive/non-zero checks
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
    #Prompt user to input parameters for the Hopalong Attractor
    a = get_validated_input('Enter a float value for "a": ', float)
    b = get_validated_input('Enter a float value for "b": ', float)
    c = get_validated_input('Enter a float value for "c": ', float)
    num = get_validated_input('Enter a positive integer value for "num": ', int, check_non_zero=True, check_positive=True)
    params = {'a': a, 'b': b, 'c': c, 'num': num}
    return a, b, c, num, params


@njit
def compute_trajectory(a, b, c, num):
    """
    njit is an alias for nopython=True
    Computes the trajectory points of the Hopalong Attractor
    """
    points = np.zeros((num, 2), dtype=np.float32)
    x = y = 0.0

    for i in range(num):
        points[i] = x, y
        xx, yy = y - copysign(1.0, x) * sqrt(fabs(b * x - c)), a - x
        # signum function respecting the behavior of floating point numbers according to IEEE 754 (signed zero)
        x, y = xx, yy

    return points


@njit
def populate_image(image, px, py):
    for i in range(len(px)):
        # populate image array, respect the row-column (y-x) indexing
        image[py[i], px[i]] += 1


def generate_trajectory_image(points, image_size):
    # Generates an image array with the mapped trajectory points
    img_width, img_height = image_size
    image = np.zeros((img_height, img_width), dtype=np.uint16)

    min_x, max_x = np.min(points[:, 0]), np.max(points[:, 0])
    min_y, max_y = np.min(points[:, 1]), np.max(points[:, 1])

    # map trajectory points to image pixel coordinates
    px = ((points[:, 0] - min_x) / (max_x - min_x)
          * (img_width - 1)).astype(np.uint16)
    py = ((points[:, 1] - min_y) / (max_y - min_y)
          * (img_height - 1)).astype(np.uint16)

    extents = [min_x, max_x, min_y, max_y]
    
    populate_image(image, px, py)

    return image, extents


def render_trajectory_image(img, extents, params, color_map):
    # Renders the trajectory of the Hopalong Attractor as an image
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(1, 1, 1, aspect='auto')
    # origin="lower" align according cartesian coordinates
    ax.imshow(img, origin="lower", cmap=color_map, extent=extents)
    ax.set_title(
        "Hopalong Attractor@ratwolf@2024\nParams: a={a}, b={b}, c={c}, num={num:_}".format(**params))
    #plt.show()
    plt.pause(1)
    plt.close(fig)


def main(image_size=(1000, 1000), color_map='hot'):
    """
    Generate Hopalong Attractor: 
    Get user inputs, compute hopalong trajectory, generate and render trajectory image.
    """
    try:
        a, b, c, num, params = get_attractor_parameters()

        # Start the time measurement
        start_time = time.process_time()

        points = compute_trajectory(a, b, c, num)
        img, extents = generate_trajectory_image(points, image_size)
        render_trajectory_image(img, extents, params, color_map)

        # End the time measurement
        end_time = time.process_time()

        # Calculate the CPU user and system time
        cpu_sys_time_used = end_time - start_time

        print(f'CPU User&System time: {cpu_sys_time_used:.2f} seconds')

    except Exception as e:
        print(f"An error occurred: {e}")

"""Main execution"""
if __name__ == "__main__":
    main()
