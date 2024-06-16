import matplotlib.pyplot as plt
import numpy as np
from numba import jit

# Use TkAgg backend for matplotlib
import matplotlib
matplotlib.use('TkAgg')


@jit(nopython=True)
def generate_hopalong_attractor_points(num, a, b, c):
   # generatedhopalong points array of shape (num, 2)
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

    px = ((points[:, 0] - min_x) / (max_x - min_x)
          * (img_width - 1)).astype(np.int32)
    py = ((points[:, 1] - min_y) / (max_y - min_y)
          * (img_height - 1)).astype(np.int32)

    return px, py


@jit(nopython=True)
def count_pixel_hits(img, px, py):
    # Calculate the hit counts for each pixel in the image
    for px_i, py_i in zip(px, py):
        img[py_i, px_i] += 1
    return img


def plot_hopalong_attractor(ax, img, colormap, extents, params):
    # plot the hopalong attractor image
    ax.imshow(img, cmap=colormap, origin='lower', extent=extents)
    ax.set_title(
        "Hopalong Attractor\nParams: a={a}, b={b}, c={c}, num={num:_}".format(**params))


def plot_hit_counts(ax, img, scale='log'):
    # plot the hit counts distribution
    hit, count = np.unique(img[img != 0], return_counts=True)
    max_count_index = np.argmax(count)
    hit_for_max_count = hit[max_count_index]
    hit_pixel = sum(j for i, j in zip(hit, count))
    img_points = np.prod(img.shape)
    hit_ratio = '{:02.2f}'.format(hit_pixel / img_points * 100)

    ax.plot(hit, count, color="red")

    ax.set_xlabel('# of hits (n)')
    ax.set_ylabel('# of pixels hit n-times')
    ax.set_title(f'Distribution of pixel hit count. \n {hit_pixel} pixels out of {img_points} image pixels = {
                 hit_ratio}% have been hit. \n The highest number of pixels with the same number of hits is {np.max(count)} with {hit_for_max_count} hits')
    ax.set_xscale(scale)
    ax.set_xlim(left=1)
    ax.set_ylim(bottom=1)
    ax.set_facecolor("black")


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
            print(f"Invalid input. Please enter a valid {
                  input_type.__name__} value.")


def plot_attractor_with_hit_count_distribution(points, a, b, c, num, image_size):
    color_map = 'hot'

    min_x, max_x = np.min(points[:, 0]), np.max(points[:, 0])
    min_y, max_y = np.min(points[:, 1]), np.max(points[:, 1])

    px, py = map_points_to_pixels(
        points, image_size, min_x, max_x, min_y, max_y)
    img = count_pixel_hits(
        np.zeros(image_size, dtype=np.int32), px, py)

    extents = [min_x, max_x, min_y, max_y]
    params = {'a': a, 'b': b, 'c': c, 'num': num}

    fig = plt.figure(figsize=(18, 8))  # change the size as needed

    # Create first subplot for the hopalong attractor plot
    # 1 row, 2 columns, first plot
    ax1 = fig.add_subplot(1, 2, 1, aspect='auto')
    plot_hopalong_attractor(ax1, img, color_map, extents, params)

    # Create second subplot for the hit counts distribution plot
    # 1 row, 2 columns, second plot
    ax2 = fig.add_subplot(1, 2, 2, aspect='auto')
    plot_hit_counts(ax2, img)

    plt.show()


def generate_and_plot_hopalong(num, a, b, c, image_size):
    # Computes and plots the Hopalong attractor.
    points = generate_hopalong_attractor_points(
        num, a, b, c).astype(np.float32)
    plot_attractor_with_hit_count_distribution(
        points, a, b, c, num, image_size)


def main():
   # Main function to run the Hopalong attractor generation.
    image_size = 1000, 1000

    a = get_validated_input(
        'Enter a non-zero float value for "a": ', float, check_non_zero=True)
    b = get_validated_input('Enter a float value for "b": ', float)
    c = get_validated_input('Enter a float value for "c": ', float)
    num = get_validated_input(
        'Enter an integer value for "num": ', int)

    generate_and_plot_hopalong(num, a, b, c, image_size)


if __name__ == "__main__":
    main()
