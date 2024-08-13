import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

import numpy as np
from numba import njit
from math import copysign, sqrt, fabs


def get_validated_input(prompt, input_type=float, check_non_zero=False, check_positive=False, min_value=None):
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
            if min_value is not None and value < min_value:
                print(f"Invalid input. The value must be at least {min_value}.")
                continue
            return value
        except ValueError:
            print(f"Invalid input. Please enter a valid {input_type.__name__} value.")


def get_attractor_parameters():
    a = get_validated_input('Enter a float value for "a": ', float)
    b = get_validated_input('Enter a float value for "b": ', float)
    while True:
        c = get_validated_input('Enter a float value for "c": ', float)
        if a == 0 and b == 0 and c == 0:
            print("Invalid combination of parameters (a = 0, b = 0, c = 0). Please enter different values.")
        else:
            break
    num = get_validated_input('Enter a positive integer value for "num": ',
                              int, check_non_zero=True, check_positive=True, min_value=10)
    return {'a': a, 'b': b, 'c': c, 'num': num}


@njit
def compute_trajectory_extents(a, b, c, num):
    # Compute the x and y extents of the Hopalong attractor trajectory.
    x = y = np.float64(0)
    min_x = min_y = np.inf
    max_x = max_y = -np.inf
    for _ in range(num):
        min_x = min(min_x, x)
        max_x = max(max_x, x)
        min_y = min(min_y, y)
        max_y = max(max_y, y)
        # signum function (copysign)respecting the behavior of floating point numbers according to IEEE 754 (signed zero)
        xx, yy = y - copysign(1.0, x) * sqrt(fabs(b * x - c)), a - x 
        x, y = xx, yy
    return min_x, max_x, min_y, max_y
# Dummy compilation call for compute_trajectory_extents
_ = compute_trajectory_extents(1.0, 1.0, 1.0, 10)


@njit
def compute_trajectory_and_image(a, b, c, num, extents, image_size):
    # Compute the trajectory and populate the image with trajectory points
    image = np.zeros(image_size, dtype=np.uint64)
    
    # pre-compute imsge scale factors
    min_x, max_x, min_y, max_y = extents
    scale_x = (image_size[0] - 1) / (max_x - min_x)
    scale_y = (image_size[1] - 1) / (max_y - min_y)
    
    x = y = np.float64(0)
    
    for _ in range(num):
        # map trajectory points to image pixel coordinates
        px = np.uint64((x - min_x) * scale_x)
        py = np.uint64((y - min_y) * scale_y)
        # populate the image
        image[py, px] += 1  # respecting row/column convention

        # Update the trajectory
        xx, yy = y - copysign(1.0, x) * sqrt(fabs(b * x - c)), a - x
        x, y = xx, yy
    return image
# Dummy compilation call for compute_trajectory_and_image
_ = compute_trajectory_and_image(1.0, 1.0, 1.0, 10, (0, 1, 0, 1), (1, 1))


def render_trajectory_image(image, extents, params, color_map):
    # Render the trajectory image
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(1, 1, 1, aspect='auto')
    # origin="lower" align according cartesian coordinates
    ax.imshow(image, origin="lower", cmap=color_map, extent=extents)
    ax.set_title("Hopalong Attractor@ratwolf@2024\nParams: a={a}, b={b}, c={c}, num={num:_}".format(**params))

    plt.show()


def main(image_size=(1000, 1000), color_map='hot'):
    # Main execution process
    try:
        params = get_attractor_parameters()
        extents = compute_trajectory_extents(params['a'], params['b'], params['c'], params['num'])
        image = compute_trajectory_and_image(params['a'], params['b'], params['c'], params['num'], extents, image_size)
        render_trajectory_image(image, extents, params, color_map)
    except Exception as e:
        print(f"An error occurred: {e}")


# Main execution
if __name__ == "__main__":
    main()

