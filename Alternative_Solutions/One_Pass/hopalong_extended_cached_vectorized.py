"""Use TkAgg backend"""
import matplotlib; matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

import numpy as np
from numba import njit, prange
from math import copysign, sqrt, fabs


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
    
    Remark: Parallel options cannot be used here due to the cross-iteration dependency.
    points[i+1] cannot be calculated without first computing points[i]
    """
    points = np.zeros((num, 2), dtype=np.float32)
    x = y = np.float32(0)

    for i in range(num):
        points[i] = x, y
        xx, yy = y - copysign(1.0, x) * sqrt(fabs(b * x - c)), a - x
        """signum function respecting the behavior of floating point numbers according to IEEE 754 (signed zero)"""
        x, y = xx, yy

    return points

@njit
def populate_image(image, px, py):
    """use of prange for parallel loop"""
    for i in prange(len(px)):
        """populate image array, respect the row-column (y-x) indexing"""
        image[py[i], px[i]] += 1


@njit
def generate_trajectory_image(points, image_size):
    """Generates an image array with the mapped trajectory points"""
    img_width, img_height = image_size
    image = np.zeros((img_height, img_width), dtype=np.uint16)

    min_x, max_x = np.min(points[:, 0]), np.max(points[:, 0])
    min_y, max_y = np.min(points[:, 1]), np.max(points[:, 1])

    """map trajectory points to image pixel coordinates"""
    px = ((points[:, 0] - min_x) / (max_x - min_x)
          * (img_width - 1)).astype(np.uint16)
    py = ((points[:, 1] - min_y) / (max_y - min_y)
          * (img_height - 1)).astype(np.uint16)

    extents = [min_x, max_x, min_y, max_y]

    populate_image(image, px, py)

    return image, extents


def render_trajectory_image(ax, img, extents, params, color_map):
    """
    Renders the trajectory of the Hopalong Attractor as an image
    origin="lower" align according cartesian coordinates
    """
    ax.imshow(img, origin="lower", cmap=color_map, extent=extents)
    ax.set_title(
        "Hopalong Attractor@ratwolf@2024\nParams: a={a}, b={b}, c={c}, num={num:_}".format(**params))

    """Add Cartesian labels"""
    ax.set_xlabel('X (Cartesian)')
    ax.set_ylabel('Y (Cartesian)')


def calculate_hit_metrics(img):
    """Analyze and summarize hit metrics from the hopalong trajectory image"""
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


def plot_hit_metrics(ax, hit_metrics, scale='log'):
    """Visualize the distribution of hit counts on pixels in the hopalong trajectory image"""
    ax.plot(hit_metrics["hit"], hit_metrics["count"], 'o-',
            color="navy", markersize=1, linewidth=0.6)
    ax.set_xlabel('# of hits (n)')
    ax.set_ylabel('# of pixels hit n-times')

    title_text = (
        f'Distribution of pixel hit count. \n'
        f'{hit_metrics["hit_pixel"]} pixels out of {hit_metrics["img_points"]} image pixels = {
            hit_metrics["hit_ratio"]}% have been hit. \n'
        f'The highest number of pixels with the same number of hits is {
            np.max(hit_metrics["count"])} with {hit_metrics["hit_for_max_count"]} hits. \n'
        f'The highest number of hits is {np.max(hit_metrics["hit"])} with {hit_metrics["count_for_max_hit"]} pixels hit')

    ax.set_title(title_text, fontsize=10)
    ax.set_xscale(scale)
    ax.set_yscale(scale)
    """x coordinate: Better resolution and display of values at the left end of the scale"""
    ax.set_xlim(left=0.9)
    """y coordinate: Better resolution and display of values at the bottom end of the scale"""
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
    """
    Generate Hopalong Attractor and hit metrics: 
    Get user inputs, compute hopalong trajectory, generate trajectory image
    Calculate hit metrics, visualize trajectory image and hit metrics
    """
    try:
        a, b, c, num, params = get_attractor_parameters()
        points = compute_trajectory(a, b, c, num)
        img, extents = generate_trajectory_image(points, image_size)
        hit_metrics = calculate_hit_metrics(img)
        visualize_trajectory_image_and_hit_metrics(
            img, extents, params, color_map, hit_metrics)
    except Exception as e:
        print(f"An error occurred: {e}")


"""Main execution"""
if __name__ == "__main__":
    main()
