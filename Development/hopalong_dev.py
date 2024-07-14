## Use TkAgg backend
import matplotlib
matplotlib.use('TkAgg')

import matplotlib.pyplot as plt                                                                      
import numpy as np
from math import copysign, sqrt, fabs
from numba import njit, prange, float32, uint32
from numba.types import Tuple


def get_user_inputs():
    # Request and validate user input with specified constraints
    def get_validated_input(prompt, input_type=float, check_non_zero=False, check_positive=False):
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
    
    a = get_validated_input('Enter a float value for "a": ', float32)
    b = get_validated_input('Enter a float value for "b": ', float32)
    c = get_validated_input('Enter a float value for "c": ', float32)
    num = get_validated_input('Enter a positive integer value for "num": ', uint32, check_non_zero=True, check_positive=True)
    params = {'a': a, 'b': b, 'c': c, 'num': num}

    return float32(a), float32(b), float32(c), uint32(num), params
    

@njit(float32[:,:](float32, float32, float32, uint32))
# support numba by explicit function signature (expected types)
def compute_trajectory(a, b, c, num):
    # Computes the trajectory points of the Hopalong Attractor
    """
    Remark: Parallel options cannot be used here due to the cross-iteration dependency.
    points[i+1] cannot be calculated without first computing points[i]
    """
    points = np.zeros((num, 2), dtype=np.float32)
    x = float32(0.0)
    y = float32(0.0)

    for i in range(num):
        points[i] = x, y
        xx, yy = y - copysign(1.0, x) * sqrt(fabs(b * x - c)), a - x
        # signum function respecting the behavior of floating point numbers according to IEEE 754 (signed zero)
        x, y = xx, yy

    return points


@njit(Tuple((uint32[:,:], float32[:]))(float32[:,:], uint32, uint32), parallel=True)
def generate_trajectory_image(points, img_width, img_height):
    # Generates an image array with the mapped trajectory points
    #img_width, img_height = image_size
    image = np.zeros((img_height, img_width), dtype=np.uint32)

    min_x, max_x = np.min(points[:, 0]), np.max(points[:, 0])
    min_y, max_y = np.min(points[:, 1]), np.max(points[:, 1])

    # map trajectory points to image pixel coordinates
    px = ((points[:, 0] - min_x) / (max_x - min_x) * (img_width - 1)).astype(np.uint32)
    py = ((points[:, 1] - min_y) / (max_y - min_y) * (img_height - 1)).astype(np.uint32)

    # use of prange for prallel loop 
    for i in prange(len(px)):
        # populate image array, respect the row-column (y-x) indexing
        image[py[i], px[i]] += 1

    #extents = [min_x, max_x, min_y, max_y]
    extents = np.array([min_x, max_x, min_y, max_y], dtype=np.float32)

    return image, extents


def render_trajectory_image(img, extents, params, color_map):
    # Renders the trajectory of the Hopalong Attractor as an image
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(1, 1, 1, aspect='auto')
    ax.imshow(img, origin="lower", cmap=color_map, extent=extents) # origin="lower" align according cartesian coordinates
    ax.set_title(
        "Hopalong Attractor@ratwolf@2024\nParams: a={a}, b={b}, c={c}, num={num:_}".format(**params))
    plt.show()

   
def main(image_width=uint32(1000), image_height=uint32(1000), color_map='hot'):
    # Generate Hopalong Attractor: Get user inputs, compute hopalong trajectory, generate and render trajectory image.

    # dummy (pre-)compilation of @njit decorated functions
    #_ = compute_trajectory(0.0, 0.0, 0.0, 1) 
    #_ = generate_trajectory_image(np.zeros((1, 2), dtype=np.float32), (1, 1))

    a, b, c, num, params = get_user_inputs()

    points = compute_trajectory(a, b, c, num)
    
    img, extents = generate_trajectory_image(points, image_width, image_height)

    render_trajectory_image(img, extents, params, color_map)


if __name__ == "__main__":
    main()
