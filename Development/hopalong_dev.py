import matplotlib
matplotlib.use('TkAgg')

from typing import Tuple, Dict
import numpy as np
from math import copysign, sqrt, fabs
from numba import njit, prange
import matplotlib.pyplot as plt



def get_user_inputs() -> Tuple[float, float, float, int, Dict[str, float]]:
    # Request and validate user input with specified constraints
    def get_validated_input(prompt: str, input_type: type = float, check_non_zero: bool = False, check_positive: bool = False) -> float:
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
                print(f"Invalid input. Please enter a valid {
                      input_type.__name__} value.")

    a = get_validated_input('Enter a float value for "a": ', float)
    b = get_validated_input('Enter a float value for "b": ', float)
    c = get_validated_input('Enter a float value for "c": ', float)
    num = get_validated_input('Enter a positive integer value for "num": ',
                              int, check_non_zero=True, check_positive=True)
    params = {'a': a, 'b': b, 'c': c, 'num': num}

    return a, b, c, num, params


@njit
def compute_trajectory(a: float, b: float, c: float, num: int) -> np.ndarray:
    # Computes the trajectory points of the Hopalong Attractor
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
def generate_trajectory_image(points: np.ndarray, image_size: Tuple[int, int]) -> Tuple[np.ndarray, list]:
    # Generates an image array with the mapped trajectory points
    img_width, img_height = image_size
    image = np.zeros((img_height, img_width), dtype=np.uint32)

    min_x, max_x = np.min(points[:, 0]), np.max(points[:, 0])
    min_y, max_y = np.min(points[:, 1]), np.max(points[:, 1])

    # map trajectory points to image pixel coordinates
    px = ((points[:, 0] - min_x) / (max_x - min_x)
          * (img_width - 1)).astype(np.uint16)
    py = ((points[:, 1] - min_y) / (max_y - min_y)
          * (img_height - 1)).astype(np.uint16)

    # use of prange for parallel loop
    for i in prange(len(px)):
        # populate image array, respect the row-column (y-x) indexing
        image[py[i], px[i]] += 1

    extents = [min_x, max_x, min_y, max_y]

    return image, extents


def render_trajectory_image(img: np.ndarray, extents: list, params: Dict[str, float], color_map: str) -> None:
    # Renders the trajectory of the Hopalong Attractor as an image
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(1, 1, 1, aspect='auto')
    # origin="lower" align according cartesian coordinates
    ax.imshow(img, origin="lower", cmap=color_map, extent=extents)
    ax.set_title(
        "Hopalong Attractor@ratwolf@2024\nParams: a={a}, b={b}, c={c}, num={num:_}".format(**params))

    plt.show()


def main(image_size: Tuple[int, int] = (1000, 1000), color_map: str = 'hot') -> None:
    # Generate Hopalong Attractor: Get user inputs, compute hopalong trajectory, generate and render trajectory image.

    a, b, c, num, params = get_user_inputs()

    points = compute_trajectory(a, b, c, num)

    img, extents = generate_trajectory_image(points, image_size)

    render_trajectory_image(img, extents, params, color_map)


if __name__ == "__main__":
    main()
