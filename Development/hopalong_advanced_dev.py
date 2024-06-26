# Use TkAgg backend for matplotlib
import matplotlib
matplotlib.use('TkAgg')

import matplotlib.pyplot as plt
import numpy as np
from numba import njit, prange

@njit
def custom_sign(x):
    """
    for floating point strictly according IEEE 754 (e.g. like implemented in Rust)
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
def calculate_attractor_points(a, b, c, num):
    # # Compute points of the hopalong attractor
    """
    Remark: Parallel options cannot be used here due to the cross-iteration dependency.
    points[i+1] cannot be calculated without first computing points[i]
    """
    points = np.zeros((num, 2), dtype=np.float32)
    x = y = 0.0

    for i in range(num):

        points[i] = x, y
        xx, yy = y - custom_sign(x) * np.sqrt(abs(b * x - c)), a - x
        x, y = xx, yy

    return points


@njit(parallel=True)
def map_attractor_points_to_image_pixels(points, image_size):
    # map attractor points to corresponding image pixel coordinates
    img_width, img_height = image_size

    min_x, max_x = np.min(points[:, 0]), np.max(points[:, 0])
    min_y, max_y = np.min(points[:, 1]), np.max(points[:, 1])

    px = ((points[:, 0] - min_x) / (max_x - min_x) * (img_width - 1)).astype(np.int32) 
    py = ((points[:, 1] - min_y) / (max_y - min_y) * (img_height - 1)).astype(np.int32)

    return px, py, min_x, max_x, min_y, max_y


@njit(parallel=True)
def generate_image_and_pixel_counts(img, px, py):
# Populate image array and aggregate hit counts per pixel from attractor points
    
    # use prange for parallel loop
    for i in prange(len(px)):
        img[px[i], py[i]] += 1

    return img


def plot_attractor(ax, img, colormap, extents, params):
    # plot image and render the hopalong attractor on a given axis with the specified colormap
    ax.imshow(img, cmap=colormap, origin='lower', extent=extents)
    ax.set_title("Hopalong Attractor@ratwolf@2024\nParams: a={a}, b={b}, c={c}, num={num:_}".format(**params))


def calculate_hit_metrics(img):
    # Analyze and summarize hit metrics from the attractor image
    hit, count = np.unique(img[img > 0], return_counts=True)
    max_count_index = np.argmax(count)
    hit_for_max_count = hit[max_count_index]
    max_hit_index = np.argmax(hit)
    count_for_max_hit = count[max_hit_index]
    hit_pixel = np.sum(count)
    img_pixels = np.prod(img.shape)
    hit_ratio = '{:02.2f}'.format(hit_pixel / img_pixels * 100)

    hit_metrics = {
        "hit": hit,
        "count": count,
        "hit_for_max_count": hit_for_max_count,
        "count_for_max_hit": count_for_max_hit,
        "hit_pixel": hit_pixel,
        "img_points": img_pixels,
        "hit_ratio": hit_ratio
    }

    return hit_metrics


def plot_hit_metrics(ax, hit_metrics, scale='log'):
    # Visualize the distribution of hit counts on pixels in the attractor image
    ax.plot(hit_metrics["hit"], hit_metrics["count"], 'o-', color="navy", markersize=1,linewidth=0.6)
    ax.set_xlabel('# of hits (n)')
    ax.set_ylabel('# of pixels hit n-times')

    title_text = (
        f'Distribution of pixel hit count. \n'
        f'{hit_metrics["hit_pixel"]} pixels out of {hit_metrics["img_points"]} image pixels = {hit_metrics["hit_ratio"]}% have been hit. \n'
        f'The highest number of pixels with the same number of hits is {np.max(hit_metrics["count"])} with {hit_metrics["hit_for_max_count"]} hits. \n'
        f'The highest number of hits is {np.max(hit_metrics["hit"])} with {hit_metrics["count_for_max_hit"]} pixels hit.')
    
    ax.set_title(title_text, fontsize=10)
    ax.set_xscale(scale)
    ax.set_yscale(scale)
    ax.set_xlim(left=0.9)   # x coordinate: Better resolution and display of values ​​at the left end of the scale
    ax.set_ylim(bottom=0.9) # y coordinate: Better resolution and display of values ​​at the bottom end of the scale
    ax.set_facecolor("lightgrey")
    ax.grid(True, which="both")


def prepare_plot_data(points, a, b, c, num, image_size):
    # Process the attractor points, hit metrics and prepare data for plotting
    px, py, min_x, max_x, min_y, max_y = map_attractor_points_to_image_pixels(points, image_size)
    img = generate_image_and_pixel_counts(np.zeros(image_size, dtype=np.int32), px, py)
    hit_metrics = calculate_hit_metrics(img) 

    extents = [min_x, max_x, min_y, max_y]
    params = {'a': a, 'b': b, 'c': c, 'num': num}
    
    return img, extents, params, hit_metrics  
    

def create_plots(img, extents, params, hit_metrics, color_map):    
    # Generate attractor and hit metrics plots
    fig = plt.figure(figsize=(18, 8))

    ax1 = fig.add_subplot(1, 2, 1, aspect='auto')
    plot_attractor(ax1, img, color_map, extents, params)
   
    ax2 = fig.add_subplot(1, 2, 2, aspect='auto')
    plot_hit_metrics(ax2, hit_metrics)

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
    # Collect input parameters from the user for attractor generation
    a = get_validated_input('Enter a non-zero float value for "a": ')#, float, check_non_zero=True)
    b = get_validated_input('Enter a float value for "b": ', float)
    c = get_validated_input('Enter a float value for "c": ', float)
    num = get_validated_input('Enter an integer value for "num": ', int, check_non_zero=True)

    return a, b, c, num


def generate_and_plot_attractor_and_statistics(a, b, c , num, image_size, color_map):
    # Generate and process attractor points, compute metrics, and create visualizations
    points = calculate_attractor_points(a, b, c, num)
    img, extents, params, hit_metrics = prepare_plot_data(points, a, b, c, num, image_size)
    create_plots(img, extents, params, hit_metrics, color_map)


def main():  
    #Entry point: Coordinate user input, processing of attractor points and statistics and visualization generation
    image_size = 1000, 1000
    color_map = 'hot'

    a, b, c, num = get_user_inputs()

    generate_and_plot_attractor_and_statistics(a, b, c , num, image_size, color_map)


if __name__ == "__main__":
    main()


