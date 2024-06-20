# Use TkAgg backend for matplotlib
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
from numba import njit


@njit
def generate_hopalong_attractor_points(num, a, b, c):
    """
    Generates Hopalong attractor points of given shape.
    
    Args:
        num (int): The number of points to generate.
        a (float): Parameter 'a' as part of Hopalong sequence definition.
        b (float): Parameter 'b' as part of Hopalong sequence definition.
        c (float): Parameter 'c' as part of Hopalong sequence definition.
        
    Returns:
        points (np.array): A NumPy array of Hopalong points with shape (num, 2).
    """
   
    points = np.zeros((num, 2), dtype=np.float32)
    x, y = 0.0, 0.0

    for i in range(num):

        points[i] = x, y
        xx, yy = y - np.sign(x) * np.sqrt(abs(b * x - c)), a - x
        x, y = xx, yy

    return points


@njit(parallel=True)
def map_attractor_points_to_image_pixels(points, image_size, min_x, max_x, min_y, max_y):
    """
    Maps Hopalong attractor points to image pixel locations.
    
    Args:
        points (np.array): A NumPy array of Hopalong points.
        image_size (tuple<int>): A tuple which defines image width and height.
        min_x (float): Minimum x coordinate of all points.
        max_x (float): Maximum x coordinate of all points.
        min_y (float): Minimum y coordinate of all points.
        max_y (float): Maximum y coordinate of all points.
        
    Returns:
        px (np.array<int>): Pixel locations along x-axis.
        py (np.array<int>): Pixel locations along y-axis.
    """
    img_width, img_height = image_size

    px = ((points[:, 0] - min_x) / (max_x - min_x)* (img_width - 1)).astype(np.int32)
    py = ((points[:, 1] - min_y) / (max_y - min_y)* (img_height - 1)).astype(np.int32)

    return px, py


@njit
def generate_image_and_pixel_counts(img, px, py):
    """
    Populates the image array with hit counts for each pixel.
    
    Args:
        img (np.array): A NumPy array representing the image.
        px (np.array<int>): Pixel locations along x-axis.
        py (np.array<int>): Pixel locations along y-axis.
        
    Returns:
        img (np.array): Modified image array.
    """
    for px_i, py_i in zip(px, py):
        img[py_i, px_i] += 1

    return img


def plot_hopalong_attractor(ax, img, colormap, extents, params):
    """
    Plots the Hopalong Attractor image.
    
    Args:
        ax (axes object): Matplotlib Axes on which the image will be drawn.
        img (np.array): A NumPy array representing the image.
        colormap (str): A colormap recognized by matplotlib to use for the image.
        extents (list<float>): List of scalar values (min_x, max_x, min_y, max_y).
        params (dict): The parameters ('a', 'b', 'c', 'num') used in image.
        
    Returns:
        None
    """
    ax.imshow(img, cmap=colormap, origin='lower', extent=extents)
    ax.set_title("Hopalong Attractor@ratwolf@2024\nParams: a={a}, b={b}, c={c}, num={num:_}".format(**params))


def calculate_pixel_hit_metrics(img):
    """
    Calculates the hit metrics.
    
    Args:
        img (np.array): A NumPy array representing the image.
        
    Returns:
        hit_metrics (dict): A dictionary containing various hit metrics.
    """
    hit, count = np.unique(img[img != 0], return_counts=True)
    max_count_index = np.argmax(count)
    hit_for_max_count = hit[max_count_index]
    max_hit_index = np.argmax(hit)
    count_for_max_hit = count[max_hit_index]
    hit_pixel = sum(j for i, j in zip(hit, count))
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
    """
    Plots the hit counts distribution.
    
    Args:
        ax (axes object): Matplotlib Axes on which the plot will be drawn.
        hit_metrics (dict): A dictionary containing various hit metrics.
        scale (str): The scale to be used for the plot.
        
    Returns:
        None
    """
    ax.plot(hit_metrics["hit"], hit_metrics["count"],color="navy", linewidth=0.6)
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
    ax.set_xlim(left=1)
    ax.set_ylim(bottom=1)
    ax.set_facecolor("lightgrey")
    ax.grid(True, which="both")


def get_validated_input(prompt, input_type=float, check_non_zero=False):
    """
    Validates user input.
    
    Args:
        prompt (str): The prompt to display for user input.
        input_type (type): The required type of user input. Default is float.
        check_non_zero (bool): Whether to check if the value is non-zero. Default is False.
        
    Returns:
        value: The validated user input value.
    """ 
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


def prepare_plot_data(points, a, b, c, num, image_size):
    """
    Processes the attractor points, calculates hit metrics, and prepares data for plotting.
    
    Args:
        points (np.array): A NumPy array of Hopalong points.
        a (float): Parameter 'a'.
        b (float): Parameter 'b'.
        c (float): Parameter 'c'.
        num (int): The number of points.
        image_size (tuple<int>): The size of the image.
        
    Returns:
        tuple: A tuple containing processed image, extents, params, and hit metrics.
    """
    min_x, max_x = np.min(points[:, 0]), np.max(points[:, 0])
    min_y, max_y = np.min(points[:, 1]), np.max(points[:, 1])
    extents = [min_x, max_x, min_y, max_y]
    params = {'a': a, 'b': b, 'c': c, 'num': num}
    px, py = map_attractor_points_to_image_pixels(points, image_size, min_x, max_x, min_y, max_y)
    img = generate_image_and_pixel_counts(np.zeros(image_size, dtype=np.int32), px, py)
    hit_metrics = calculate_pixel_hit_metrics(img)  
    
    return img, extents, params, hit_metrics  


def create_plots(img, extents, params, hit_metrics, color_map):  
    """
    Creates all the plots.
    
    Args:
        img (np.array): A NumPy array representing the image.
        extents (list<float>): The extents of the plot.
        params (dict): Parameters used in the plot.
        hit_metrics (dict): Hit metrics for the plot.
        color_map (str): Color map used in the plot.
        
    Returns:
        None
    """
    fig = plt.figure(figsize=(18, 8))
    ax1 = fig.add_subplot(1, 2, 1, aspect='auto')
    plot_hopalong_attractor(ax1, img, color_map, extents, params)
    ax2 = fig.add_subplot(1, 2, 2, aspect='auto')
    plot_hit_metrics(ax2, hit_metrics)
    plt.show()


def run_hopalong_analysis(num, a, b, c, image_size, color_map):
    """
    coordinates the process execution
    
    Args:
        num (int): The number of points.
        a (float): Parameter 'a'.
        b (float): Parameter 'b'.
        c (float): Parameter 'c'.
        image_size (tuple<int>): The size of the image.
        color_map (str): The color map to use.
        
    Returns:
        None
    """
    points = generate_hopalong_attractor_points(num, a, b, c)
    img, extents, params, hit_metrics = prepare_plot_data(points, a, b, c, num, image_size)
    create_plots(img, extents, params, hit_metrics, color_map)


def main():
    """
    Main function to define image_size and color_map, prompt for user input, and trigger the program execution.
    
    Returns:
        None
    """

    image_size = 1000, 1000
    color_map = 'hot'

    a = get_validated_input('Enter a non-zero float value for "a": ', float, check_non_zero=True)
    b = get_validated_input('Enter a float value for "b": ', float)
    c = get_validated_input('Enter a float value for "c": ', float)
    num = get_validated_input('Enter an integer value for "num": ', int)

    run_hopalong_analysis(num, a, b, c, image_size, color_map) 

if __name__ == "__main__":
    main()
