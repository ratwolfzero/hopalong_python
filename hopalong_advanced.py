import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

import numpy as np
from numba import njit
from math import copysign, sqrt, fabs


def get_validated_input(prompt, input_type=float, check_positive_non_zero=False, min_value=None):
    # Prompt for and return user input validated by type and positive/non-zero checks
    while True:
        user_input = input(prompt)
        try:
            value = input_type(user_input)
            if check_positive_non_zero and value <= 0:
                print("Invalid input. The value must be a positive non-zero number.")
                continue
            if min_value is not None and value < min_value:
                print(f"Invalid input. The value should be at least {min_value}.")
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
    num = get_validated_input('Enter a positive integer value for "num": ', int, check_positive_non_zero=True, min_value=1000)
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
        # signum function respecting the behavior of floating point numbers according to IEEE 754 (signed zero)
        xx, yy = y - copysign(1.0, x) * sqrt(fabs(b * x - c)), a - x
        x, y = xx, yy
    return min_x, max_x, min_y, max_y
# Dummy compilation call for compute_trajectory_extents
_ = compute_trajectory_extents(1.0, 1.0, 1.0, 5)


@njit
def compute_trajectory_and_image(a, b, c, num, extents, image_size):
    # Compute the trajectory and populate the image with trajectory points
    img_width, img_height = image_size
    image = np.zeros((img_height, img_width), dtype=np.uint64)
    
    # pre-compute imsge scale factors
    min_x, max_x, min_y, max_y = extents
    scale_x = (img_width - 1) / (max_x - min_x)
    scale_y = (img_height - 1) / (max_y - min_y)
    
    x = y = np.float64(0)
    
    for _ in range(num):
        # map trajectory points to image pixel coordinates
        px = np.uint64((x - min_x) * scale_x)
        py = np.uint64((y - min_y) * scale_y)
        #populate the image
        image[py, px] += 1 # respecting row/column convention

        # update the trajectory
        xx, yy = y - copysign(1.0, x) * sqrt(fabs(b * x - c)), a - x
        x, y = xx, yy
    return image
# Dummy compilation call for compute_trajectory_and_image
_ = compute_trajectory_and_image(1.0, 1.0, 1.0, 5, (0, 1, 0, 1), (1, 1))


def calculate_hit_metrics(img):
    hit, count = np.unique(img[img > 0], return_counts=True)
    max_count_index = np.argmax(count)
    hit_for_max_count = hit[max_count_index]
    max_hit_index = np.argmax(hit)
    count_for_max_hit = count[max_hit_index]
    hit_pixel = np.sum(count)
    img_pixels = np.prod(img.shape)
    hit_ratio = '{:.2f}'.format(hit_pixel / img_pixels * 100)

    hit_metrics = {
        "hit": hit,
        "count": count,
        "hit_for_max_count": hit_for_max_count,
        "count_for_max_hit": count_for_max_hit,
        "hit_pixel": hit_pixel,
        "img_points": img_pixels,
        "hit_ratio": hit_ratio,
    }
    return hit_metrics


def render_trajectory_image(ax, img, extents, params, color_map):
    ax.imshow(img, origin="lower", cmap=color_map, extent=extents)
    ax.set_title(
        "Hopalong Attractor@ratwolf@2024\nParams: a={a}, b={b}, c={c}, num={num:_}".format(**params))
    ax.set_xlabel('X (Cartesian)')
    ax.set_ylabel('Y (Cartesian)')


def plot_hit_metrics(ax, hit_metrics, scale='log'):
    ax.plot(hit_metrics["hit"], hit_metrics["count"], 'o-', color="navy", markersize=1, linewidth=0.6)
    ax.set_xlabel('# of hits (n)')
    ax.set_ylabel('# of pixels hit n-times')

    title_text = (
        f'Distribution of pixel hit count. \n'
        f'{hit_metrics["hit_pixel"]} pixels out of {hit_metrics["img_points"]} image pixels = {hit_metrics["hit_ratio"]}% have been hit at least one time. \n'
        f'The highest number of pixels with the same number of hits is {np.max(hit_metrics["count"])} with {hit_metrics["hit_for_max_count"]} hits. \n'
        f'The highest number of hits is {np.max(hit_metrics["hit"])} with {hit_metrics["count_for_max_hit"]} pixels hit')

    ax.set_title(title_text, fontsize=10)
    ax.set_xscale(scale)
    ax.set_yscale(scale)
    ax.set_xlim(left=0.9)
    ax.set_ylim(bottom=0.9)
    ax.set_facecolor("lightgrey")
    ax.grid(True, which="both")


def visualize_trajectory_image_and_hit_metrics(img, extents, params, color_map, hit_metrics):
    fig = plt.figure(figsize=(18, 8))

    ax1 = fig.add_subplot(1, 2, 1, aspect='auto')
    render_trajectory_image(ax1, img, extents, params, color_map)

    ax2 = fig.add_subplot(1, 2, 2, aspect='auto')
    plot_hit_metrics(ax2, hit_metrics)

    plt.show()


def main(image_size=(1000, 1000), color_map='hot'):
    # Main execution process
    try:
        params = get_attractor_parameters()
        extents = compute_trajectory_extents(params['a'], params['b'], params['c'], params['num'])
        image = compute_trajectory_and_image(params['a'], params['b'], params['c'], params['num'], extents, image_size)
        hit_metrics = calculate_hit_metrics(image)
        visualize_trajectory_image_and_hit_metrics(image, extents, params, color_map, hit_metrics)
    except Exception as e:
        print(f"An error occurred: {e}")


# Main execution
if __name__ == "__main__":
    main()

