import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

import numpy as np
from numba import njit
from math import copysign, sqrt, fabs

import time 


def get_validated_input(prompt, input_type=float, check_non_zero=False, check_positive=False):
    # Prompt for and return user input validated by type and positive/non-zero checks
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
    # Prompt user to input parameters for the Hopalong Attractor
    params = {
        'a': get_validated_input('Enter a float value for "a": ', float),
        'b': get_validated_input('Enter a float value for "b": ', float),
        'c': get_validated_input('Enter a float value for "c": ', float),
        'num': get_validated_input('Enter a positive integer value for "num": ', int, check_non_zero=True, check_positive=True)
    }
    return params


@njit(cache=True)
def compute_full_trajectory_extents(a, b, c, num):
    # Compute the x and y extents of the Hopalong attractor trajectory. Cross iteration dependency cannot be parallelized
    x = y = np.float64(0)
    min_x = min_y = np.inf
    max_x = max_y = -np.inf
    for _ in range(num):
        min_x = min(min_x, x); max_x = max(max_x, x)
        min_y = min(min_y, y); max_y = max(max_y, y)
        xx, yy = y - copysign(1.0, x) * sqrt(fabs(b * x - c)), a - x
        x, y = xx, yy
    return min_x, max_x, min_y, max_y


def generate_chunk_sizes(num, chunk_size):
    # generator function yield sizes of chunks to process in each iteration until covering the entire range
    for i in range(0, num, chunk_size):
        current_chunk_size = min(chunk_size, num - i)
        yield current_chunk_size


@njit(cache=True)
def compute_trajectory_chunk(a, b, c, current_chunk_size, x0, y0):
    # Compute a chunk of the Hopalong trajectory. Cross iteration dependency cannot be parallelized
    points = np.zeros((current_chunk_size, 2), dtype=np.float64)
    x, y = x0, y0
    for i in range(current_chunk_size):
        points[i] = x, y
        # signum function respecting the behavior of floating point numbers according to IEEE 754 (signed zero)
        xx, yy = y - copysign(1.0, x) * sqrt(fabs(b * x - c)), a - x
        x, y = xx, yy
    return points, x, y


@njit(cache=True)
def map_trajectory_chunk_to_image(image, points, scale_x, scale_y, min_x, min_y):
    # map trajectory chunk points to image pixel locations
    n_points = points.shape[0]
    for i in range(n_points):
        px, py = np.uint64((points[i, 0] - min_x) * scale_x), np.uint64((points[i, 1] - min_y) * scale_y)
        image[py, px] += 1 # respecting row/column convention
"""
Avoiding Numpy vectorization, parallelization with Numba / Numna prange, parallel iteration with Python zip 
is obviously the fastest solution using the @njit decorator and avoids race conditions caused by prange
"""


def compute_full_trajectory_image(a, b, c, num, chunk_size, extents, image_size):
    # Calculate the full trajectory image from chunks
    image = np.zeros(image_size, dtype=np.uint64)

    min_x, max_x, min_y, max_y = extents
    scale_x, scale_y = (image_size[1] - 1) / (max_x - min_x), (image_size[0] - 1) / (max_y - min_y)

    x0 = y0 = np.float64(0)

    for current_chunk_size in generate_chunk_sizes(num, chunk_size):
        # x0 and y0 track the current trajectory state and are updated after each chunk to maintain continuity.
        points, x0 ,y0  = compute_trajectory_chunk(a, b, c, current_chunk_size, x0, y0)
        map_trajectory_chunk_to_image(image, points, scale_x, scale_y, min_x, min_y)

    return image


def render_full_trajectory_image(image, extents, params, color_map):
    # Render the full trajectory image
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(1, 1, 1, aspect='auto')
    # origin="lower" align according cartesian coordinates
    ax.imshow(image, origin="lower", cmap=color_map, extent=extents)
    ax.set_title("Hopalong Attractor@ratwolf@2024\nParams: a={a}, b={b}, c={c}, num={num:_}".format(**params))

    #plt.show()
    plt.pause(1)
    plt.close(fig)


def main(image_size=(1000, 1000), color_map='hot', chunk_size=1750000):
    # Execute processes to generate and render the Hopalong Attractor
    try:
        params = get_attractor_parameters()
        start_time = time.process_time()

        extents = compute_full_trajectory_extents(params['a'], params['b'], params['c'], params['num'])
        image = compute_full_trajectory_image(params['a'], params['b'], params['c'], params['num'], chunk_size, extents, image_size)
        render_full_trajectory_image(image, extents, params, color_map)

        end_time = time.process_time()
        cpu_sys_time_used = end_time - start_time
        print(f'CPU User&System time: {cpu_sys_time_used:.2f} seconds')
        
        
    except Exception as e:
        print(f"An error occurred: {e}")


# Main execution
if __name__ == "__main__":
    main()

