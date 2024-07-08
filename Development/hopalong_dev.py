# Use TkAgg backend
import matplotlib
matplotlib.use('TkAgg')

import matplotlib.pyplot as plt
import numpy as np
from numba import njit, prange
from math import copysign, sqrt, fabs
    

@njit
def compute_hopalong_trajectory(a, b, c, num):
    # Computes the trajectory of the Hopalong Attractor
    """
    Remark: Parallel options cannot be used here due to the cross-iteration dependency.
    points[i+1] cannot be calculated without first computing points[i]
    """
    points = np.zeros((num, 2), dtype=np.float32)
    x = y = 0.0

    for i in range(num):
        points[i] = x, y
        xx, yy = y - copysign(1.0, x) * sqrt(fabs(b * x - c)), a - x
        # signum function respecting the behavior of floating point numbers according to IEEE 754 (signed zero)
        x, y = xx, yy

    return points


@njit(parallel=True)
def generate_trajectory_image(points, image_size):
    # Generates an image array with the mapped trajectory points
    img_width, img_height = image_size
    image = np.zeros((img_height, img_width), dtype=np.uint32)

    min_x, max_x = np.min(points[:, 0]), np.max(points[:, 0])
    min_y, max_y = np.min(points[:, 1]), np.max(points[:, 1])

    px = ((points[:, 0] - min_x) / (max_x - min_x) * (img_width - 1)).astype(np.uint16)
    py = ((points[:, 1] - min_y) / (max_y - min_y) * (img_height - 1)).astype(np.uint16)

    for i in prange(len(px)):
        image[py[i], px[i]] += 1

    extents = [min_x, max_x, min_y, max_y]

    return image, extents


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
            print(f"Invalid input. Please enter a valid {input_type.__name__} value.")


def get_user_inputs():
    # Collect input parameters from the user for hopalong attractor trajectory generation
    a = get_validated_input('Enter a non-zero float value for "a": ', float)
    b = get_validated_input('Enter a float value for "b": ', float)
    c = get_validated_input('Enter a float value for "c": ', float)
    num = get_validated_input('Enter an integer value for "num": ', int, check_non_zero=True)

    return a, b, c, num
    

def main():
    # Entry point: Generate Hopalong Attractor: Compute hopalong trajectory, generate trajectory image and render trajectory imgae
 
    a, b, c, num = get_user_inputs()
    points = compute_hopalong_trajectory(a, b, c, num)

    image_size = 1000, 1000 
    img, extents = generate_trajectory_image(points, image_size)

    params = {'a': a, 'b': b, 'c': c, 'num': num} 
    color_map = 'hot'    
    render_trajectory_image(img, extents, params, color_map)


if __name__ == "__main__":
    main()