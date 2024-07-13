# Use TkAgg backend
import matplotlib
matplotlib.use('TkAgg')

import matplotlib.pyplot as plt
import numpy as np
from numba import njit, prange
from math import copysign, sqrt, fabs
    

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
    
    a = get_validated_input('Enter a float value for "a": ', float)
    b = get_validated_input('Enter a float value for "b": ', float)
    c = get_validated_input('Enter a float value for "c": ', float)
    num = get_validated_input('Enter a positive integer value for "num": ', int, check_non_zero=True, check_positive=True)
    params = {'a': a, 'b': b, 'c': c, 'num': num}

    return a, b, c, num, params


@njit('float32[:,:](float32, float32, float32, uint32)')
# support numba by explicit function signature (expected types)
def compute_trajectory(a, b, c, num):
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
def generate_trajectory_image(points, image_size):
    # Generates an image array with the mapped trajectory points
    img_width, img_height = image_size
    image = np.zeros((img_height, img_width), dtype=np.uint32)

    min_x, max_x = np.min(points[:, 0]), np.max(points[:, 0])
    min_y, max_y = np.min(points[:, 1]), np.max(points[:, 1])

    # map trajectory points to image pixel coordinates
    px = ((points[:, 0] - min_x) / (max_x - min_x) * (img_width - 1)).astype(np.uint16)
    py = ((points[:, 1] - min_y) / (max_y - min_y) * (img_height - 1)).astype(np.uint16)

    for i in prange(len(px)): # use of prange for prallel loop
        image[py[i], px[i]] += 1 # populate image array, respect the row-column (y-x) indexing

    extents = [min_x, max_x, min_y, max_y]

    return image, extents


def render_trajectory_image(ax, img, extents, params, color_map):
    # Renders the trajectory of the Hopalong Attractor as an image
    ax.imshow(img, origin="lower", cmap=color_map, extent=extents) # origin="lower" align according cartesian coordinates
    ax.set_title(
        "Hopalong Attractor@ratwolf@2024\nParams: a={a}, b={b}, c={c}, num={num:_}".format(**params))
    
    # Add Cartesian labels
    ax.set_xlabel('X (Cartesian)')
    ax.set_ylabel('Y (Cartesian)')
    

def calculate_hit_metrics(img, extents):
    # Analyze and summarize hit metrics from the hopalong trajectory image
    hit, count = np.unique(img[img > 0], return_counts=True)
    max_count_index = np.argmax(count)
    hit_for_max_count = hit[max_count_index]
    max_hit_index = np.argmax(hit)
    count_for_max_hit = count[max_hit_index]
    hit_pixel = np.sum(count)
    img_pixels = np.prod(img.shape)
    hit_ratio = '{:.2f}'.format(hit_pixel / img_pixels * 100)

     # Find all pixels with the highest hit count
    max_hit_count = np.max(img)
    max_hit_coords = np.argwhere(img == max_hit_count)
    
    # Convert image coordinates pixels with the highest hit count to Cartesian coordinates
    img_height, img_width = img.shape
    min_x, max_x, min_y, max_y = extents
    cartesian_x = min_x + (max_x - min_x) * (max_hit_coords[:, 1] / img_width)
    cartesian_y = min_y + (max_y - min_y) * (max_hit_coords[:, 0] / img_height)
    cartesian_coords = ['({:.3f}, {:.3f})'.format(x, y) for x, y in zip(cartesian_x, cartesian_y)]
        
    hit_metrics = {
        "hit": hit,
        "count": count,
        "hit_for_max_count": hit_for_max_count,
        "count_for_max_hit": count_for_max_hit,
        "hit_pixel": hit_pixel,
        "img_points": img_pixels,
        "hit_ratio": hit_ratio,
        "cartesian_coords": cartesian_coords
        }
    
    return hit_metrics


def plot_hit_metrics(ax, hit_metrics, scale='log'):
    # Visualize the distribution of hit counts on pixels in the hopalong trajectory image
    ax.plot(hit_metrics["hit"], hit_metrics["count"], 'o-', color="navy", markersize=1,linewidth=0.6)
    ax.set_xlabel('# of hits (n)')
    ax.set_ylabel('# of pixels hit n-times')

    title_text = (
        f'Distribution of pixel hit count. \n'
        f'{hit_metrics["hit_pixel"]} pixels out of {hit_metrics["img_points"]} image pixels = {hit_metrics["hit_ratio"]}% have been hit. \n'
        f'The highest number of pixels with the same number of hits is {np.max(hit_metrics["count"])} with {hit_metrics["hit_for_max_count"]} hits. \n'
        f'The highest number of hits is {np.max(hit_metrics["hit"])} with {hit_metrics["count_for_max_hit"]} pixels hit\n'
        f'Cartesian coordinates for highest #of hits:{hit_metrics["cartesian_coords"]}')
    
    ax.set_title(title_text, fontsize=10)
    ax.set_xscale(scale)
    ax.set_yscale(scale)
    ax.set_xlim(left=0.9)   # x coordinate: Better resolution and display of values ​​at the left end of the scale
    ax.set_ylim(bottom=0.9) # y coordinate: Better resolution and display of values ​​at the bottom end of the scale
    ax.set_facecolor("lightgrey")
    ax.grid(True, which="both")


def visualize_trajectory_image_and_hit_metrics(img, extents, params, color_map, hit_metrics):
    
    fig = plt.figure(figsize=(24, 8))

    ax1 = fig.add_subplot(1, 2, 1, aspect='auto')
    render_trajectory_image(ax1, img, extents, params, color_map)
    
    ax2 = fig.add_subplot(1, 2, 2, aspect='auto')
    plot_hit_metrics(ax2, hit_metrics)

    plt.show()

    
def main(image_size=(1000, 1000), color_map='hot'):
    """
    Generate Hopalong Attractor and hit metrics: Get user inputs, compute hopalong trajectory, generate trajectory image.
    Calculate hit metrics, visualize trajectory image and hit metrics
    """
    # dummy (pre-)compilation of @njit decorated functions
    _ = compute_trajectory(0.0, 0.0, 0.0, 1) 
    _ = generate_trajectory_image(np.zeros((1, 2), dtype=np.float32), (1, 1))

    a, b, c, num, params = get_user_inputs()

    points = compute_trajectory(a, b, c, num)

    img, extents = generate_trajectory_image(points, image_size)
    
    hit_metrics = calculate_hit_metrics(img, extents)

    visualize_trajectory_image_and_hit_metrics(img, extents, params, color_map, hit_metrics)


if __name__ == "__main__":
    main()