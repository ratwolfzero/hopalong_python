# Use TkAgg backend
import matplotlib; matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

import numpy as np
from numba import njit, prange
from math import copysign, sqrt, fabs


def get_user_inputs():
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

    a = get_validated_input('Enter a float value for "a": ', float)                         
    b = get_validated_input('Enter a float value for "b": ', float)
    c = get_validated_input('Enter a float value for "c": ', float)
    num = get_validated_input('Enter a positive integer value for "num": ', int, check_non_zero=True, check_positive=True)
    params = {'a': a, 'b': b, 'c': c, 'num': num}

    return a, b, c, num, params

@njit
def compute_trajectory_chunk(a, b, c, num, x0, y0):
    points = np.empty((num, 2), dtype=np.float64)
    x, y = x0, y0

    for i in range(num):
        points[i] = x, y
        xx, yy = y - copysign(1.0, x) * sqrt(fabs(b * x - c)), a - x
        x, y = xx, yy

    return points, x, y

@njit
def compute_extents(a, b, c, num):
    x = y = np.float64(0)
    min_x = min_y = np.inf
    max_x = max_y = -np.inf

    for i in range(num):
        min_x = min(min_x, x)
        max_x = max(max_x, x)
        min_y = min(min_y, y)
        max_y = max(max_y, y)
        xx, yy = y - copysign(1.0, x) * sqrt(fabs(b * x - c)), a - x
        x, y = xx, yy

    return min_x, max_x, min_y, max_y

@njit(parallel=True)
def update_image(image, points, min_x, max_x, min_y, max_y):
    img_width, img_height = image.shape[1], image.shape[0]

    px = ((points[:, 0] - min_x) / (max_x - min_x) * (img_width - 1)).astype(np.uint64)
    py = ((points[:, 1] - min_y) / (max_y - min_y) * (img_height - 1)).astype(np.uint64)

    for i in prange(len(px)):
        if 0 <= px[i] < img_width and 0 <= py[i] < img_height:
            image[py[i], px[i]] += 1

def render_trajectory_image(img, extents, params, color_map):
    fig = plt.figure(figsize=(8, 8))                                                        
    ax = fig.add_subplot(1, 1, 1, aspect='auto')
    ax.imshow(img, origin="lower", cmap=color_map, extent=extents)
    ax.set_title(
        "Hopalong Attractor@ratwolf@2024\nParams: a={a}, b={b}, c={c}, num={num:_}".format(**params))
    plt.show()

@njit
def create_chunks(num, chunk_size):
    """
    Generate indices for chunks.
    """
    for i in range(0, num, chunk_size):
        current_chunk_size = min(chunk_size, num - i)
        yield i, current_chunk_size
        
@njit
def calculate_image(a, b, c, num, chunk_size, min_x, max_x, min_y, max_y, img_height, img_width):
    """
    Calculate image from chunks.
    """
    image = np.zeros((img_height, img_width), dtype=np.uint64)
    x0 = y0 = np.float64(0)

    for i, current_chunk_size in create_chunks(num, chunk_size):
        points, x0, y0 = compute_trajectory_chunk(a, b, c, current_chunk_size, x0, y0)
        update_image(image, points, min_x, max_x, min_y, max_y)

    return image

def main(image_size=(8000, 8000), color_map='hot', chunk_size=1000000):
    """
    Generate Hopalong Attractor: 
    Get user inputs, compute hopalong trajectory, generate and render trajectory image.
    """
    a, b, c, num, params = get_user_inputs()

    img_width, img_height = image_size
    min_x, max_x, min_y, max_y = compute_extents(a, b, c, num)

    image = calculate_image(a, b, c, num, chunk_size, min_x, max_x,
                            min_y, max_y, img_height, img_width)

    render_trajectory_image(image, [min_x, max_x, min_y, max_y], params, color_map)

# Main execution
if __name__ == "__main__":
    main()