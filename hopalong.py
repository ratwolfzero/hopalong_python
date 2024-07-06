# Use TkAgg backend
import matplotlib
matplotlib.use('TkAgg')

import matplotlib.pyplot as plt
import numpy as np
from numba import njit, prange
from math import copysign


@njit
def custom_sign(x):
    """
    Custom sign function for floating point according to IEEE 754 (e.g. like implemented in Rust)
    Returns:
        1.0 if the number is positive, +0.0 or INFINITY
        -1.0 if the number is negative, -0.0 or NEG_INFINITY
        NaN if the number is NaN
    """
    if np.isnan(x):
        return np.nan
    elif x > 0 or x == 0.0:
        return 1.0
    else:
        return -1.0
    

@njit
def hopalong_trajectory_simulation(a, b, c, num):
    # Simulates the trajectory of the Hopalong Attractor
    """
    Remark: Parallel options cannot be used here due to the cross-iteration dependency.
    points[i+1] cannot be calculated without first computing points[i]
    """
    points = np.zeros((num, 2), dtype=np.float32)
    x = y = 0.0

    for i in range(num):

        points[i] = x, y
        xx, yy = y - copysign(1.0, x) * np.sqrt(abs(b * x - c)), a - x # Variant using math.copysign signum function
        #xx, yy = y - custom_sign(x) * np.sqrt(abs(b * x - c)), a - x  # Variant using custom signum function
        # xx, yy = y - np.sign(x) * np.sqrt(abs(b * x - c)), a - x     # Variant using Numpy standard signum function
        x, y = xx, yy

    return points


@njit(parallel=True)
def map_points_to_image(points, image_size):
    # Maps the trajectory points to an image grid
    img_width, img_height = image_size

    min_x, max_x = np.min(points[:, 0]), np.max(points[:, 0])
    min_y, max_y = np.min(points[:, 1]), np.max(points[:, 1])

    px = ((points[:, 0] - min_x) / (max_x - min_x) * (img_width - 1)).astype(np.uint16) 
    py = ((points[:, 1] - min_y) / (max_y - min_y) * (img_height - 1)).astype(np.uint16)

    return px, py, min_x, max_x, min_y, max_y


@njit(parallel=True)
def generate_image(img, px, py):
# Populates an image array with trajectory points. Each point gets a unique value based on the hit count
    
    # use prange for parallel loop
    for i in prange(len(px)):
        img[px[i], py[i]] += 1     # Variant: Each point gets a unique value based on the hit count
        #img[px[i], py[i]] = i + 1 # Variant: Each point gets a unique value based on the index

    return img


def prepare_plot_data(points, a, b, c, num, image_size):
    # Processes trajectory points and prepares data for visualization
    px, py, min_x, max_x, min_y, max_y = map_points_to_image(points, image_size)
    img = generate_image(np.zeros(image_size, dtype=np.uint32), px, py)

    extents = [min_x, max_x, min_y, max_y]
    params = {'a': a, 'b': b, 'c': c, 'num': num}

    return img, extents, params


def render_trajectory_image(img, extents, params, color_map):
    # Renders the trajectory of the Hopalong Attractor as an image
    plt.figure(figsize=(8, 8))
    plt.imshow(img, origin="lower", cmap=color_map, extent=extents)
    plt.title(
        "Hopalong Attractor@ratwolf@2024\nParams: a={a}, b={b}, c={c}, num={num:_}".format(**params))
    plt.show()


def get_validated_input(prompt, input_type=float, check_non_zero=False):
    # Request and validate user input with specified constraints
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
    # Collect input parameters from the user for hopalong attractor trajetory generation
    a = get_validated_input('Enter a non-zero float value for "a": ', float)
    b = get_validated_input('Enter a float value for "b": ', float)
    c = get_validated_input('Enter a float value for "c": ', float)
    num = get_validated_input('Enter an integer value for "num": ', int, check_non_zero=True)

    return a, b, c, num


def simulate_trajectory_and_render_trajectory_image(a, b, c, num, image_size, color_map):
    # simulate hopalong attractor trajetory and create visualizations
    points = hopalong_trajectory_simulation(a, b, c, num)
    img, extents, params = prepare_plot_data(points, a, b, c, num, image_size)
    render_trajectory_image(img, extents, params, color_map)


def main():
    #Entry point: Coordinate user input, processing of attractor trajectory and visualization generation

    image_size = 1000, 1000

    color_map = 'hot'     # for variant each point gets a unique value based on the hit count
    #color_map = 'inferno' # for variant each point gets a unique value based on the index
 
    a, b, c, num = get_user_inputs()

    simulate_trajectory_and_render_trajectory_image(a, b, c, num, image_size, color_map)


if __name__ == "__main__":
    main()