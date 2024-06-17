# Use TkAgg backend for matplotlib
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
from numba import jit


@jit(nopython=True)
def generate_hopalong_attractor_points(num, a, b, c):
   # generate hopalong points array of shape (num, 2)
    points = np.zeros((num, 2), dtype=np.float32)
    x, y = 0.0, 0.0

    for i in range(num):

        points[i] = x, y
        xx, yy = y - np.sign(x) * np.sqrt(abs(b * x - c)), a - x
        x, y = xx, yy

    return points


def map_points_to_pixels(points, image_size, min_x, max_x, min_y, max_y):
    # Convert hopalong points to pixel locations
    img_width, img_height = image_size

    px = ((points[:, 0] - min_x) / (max_x - min_x)* (img_width - 1)).astype(np.int32)
    py = ((points[:, 1] - min_y) / (max_y - min_y)* (img_height - 1)).astype(np.int32)

    return px, py


@jit(nopython=True)
def generate_image_and_pixel_counts(img, px, py):
    # Calculate the hit counts for each pixel in the image
    for px_i, py_i in zip(px, py):
        img[py_i, px_i] += 1
    return img


def plot_hopalong_attractor(ax, img, colormap, extents, params):
    # plot the hopalong attractor image
    ax.imshow(img, cmap=colormap, origin='lower', extent=extents)
    ax.set_title("Hopalong Attractor@ratwolf@2024\nParams: a={a}, b={b}, c={c}, num={num:_}".format(**params))


def calculate_hit_metrics(img):
    # Calculate hit metrics
    hit, count = np.unique(img[img != 0], return_counts=True)
    max_count_index = np.argmax(count)
    hit_for_max_count = hit[max_count_index]
    max_hit_index = np.argmax(hit)
    count_for_max_hit = count[max_hit_index]
    hit_pixel = sum(j for i, j in zip(hit, count))
    img_points = np.prod(img.shape)
    hit_ratio = '{:02.2f}'.format(hit_pixel / img_points * 100)

    hit_metrics = {
        "hit": hit,
        "count": count,
        "hit_for_max_count": hit_for_max_count,
        "count_for_max_hit": count_for_max_hit,
        "hit_pixel": hit_pixel,
        "img_points": img_points,
        "hit_ratio": hit_ratio
    }

    return hit_metrics


def plot_hit_counts(ax, hit_metrics, scale='log'):
    # Plot the hit counts distribution
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
    # Prompts the user for input and validates it.
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
    # returns the data necessary for plotting
    min_x, max_x = np.min(points[:, 0]), np.max(points[:, 0])
    min_y, max_y = np.min(points[:, 1]), np.max(points[:, 1])
    extents = [min_x, max_x, min_y, max_y]
    params = {'a': a, 'b': b, 'c': c, 'num': num}
    px, py = map_points_to_pixels(points, image_size, min_x, max_x, min_y, max_y)
    img = generate_image_and_pixel_counts(np.zeros(image_size, dtype=np.int32), px, py)
    hit_metrics = calculate_hit_metrics(img)  
    
    
    return img, extents, params, hit_metrics  


def create_plots(img, extents, params, hit_metrics):  
    # generates the plots
    color_map = 'hot'
    fig = plt.figure(figsize=(18, 8))
    ax1 = fig.add_subplot(1, 2, 1, aspect='auto')
    plot_hopalong_attractor(ax1, img, color_map, extents, params)
    ax2 = fig.add_subplot(1, 2, 2, aspect='auto')
    plot_hit_counts(ax2, hit_metrics)
    plt.show()


def run_hopalong_analysis(num, a, b, c, image_size):
    #coordinates the process
    points = generate_hopalong_attractor_points(num, a, b, c).astype(np.float32)
    img, extents, params, hit_metrics = prepare_plot_data(points, a, b, c, num, image_size)
    create_plots(img, extents, params, hit_metrics)

def main():
   # Main function to run the Hopalong analysis.
    image_size = 1000, 1000

    a = get_validated_input('Enter a non-zero float value for "a": ', float, check_non_zero=True)
    b = get_validated_input('Enter a float value for "b": ', float)
    c = get_validated_input('Enter a float value for "c": ', float)
    num = get_validated_input('Enter an integer value for "num": ', int)

    run_hopalong_analysis(num, a, b, c, image_size)


if __name__ == "__main__":
    main()
