import matplotlib.pyplot as plt
import numpy as np
from numba import jit

# Use TkAgg backend for matplotlib
import matplotlib
matplotlib.use('TkAgg')


@jit(nopython=True)
def hopalong_compute(num, a, b, c):
   # Computes the points for the Hopalong attractor.
    points = np.empty((num, 2), dtype=np.float32)
    x, y = 0.0, 0.0

    for i in range(num):

        points[i] = x, y
        xx, yy = y - np.sign(x) * np.sqrt(abs(b * x - c)), a - x
        x, y = xx, yy

    return points


@jit(nopython=True)
def pixels_and_hit_count(img, px, py):
    # set pixels and track their density (used for color sheme cmap)
    for px_i, py_i in zip(px, py):
        img[py_i, px_i] += 1
    return img


def hopalong_plot(points, a, b, c, num, image_size):
    # Plots the points of the Hopalong attractor.
    min_x, max_x = np.min(points[:, 0]), np.max(points[:, 0])
    min_y, max_y = np.min(points[:, 1]), np.max(points[:, 1])

    img_width, img_height = image_size
    img = np.zeros((img_height, img_width), dtype=np.int64)

    px = ((points[:, 0] - min_x) / (max_x - min_x)
          * (img_width - 1)).astype(np.int64)
    py = ((points[:, 1] - min_y) / (max_y - min_y)
          * (img_height - 1)).astype(np.int64)

    img = pixels_and_hit_count(img, px, py)
    
    plt.figure(figsize=(10, 10))
    plt.imshow(img, origin="lower", cmap='hot',
               extent=[min_x, max_x, min_y, max_y])
    plt.title(
        f"Hopalong Attractor@ratwolf2024\nParams: a={a}, b={b}, c={c}, num={num:_}")
    
    hit, count = np.unique(img[img!=0], return_counts=True)
    fig_w, fig_h = 12, 8
    plt.figure(figsize=(fig_w, fig_h))
    plt.xlabel('number of hits (n)',fontsize=12)
    plt.ylabel('number of pixels hit n-times',fontsize=12)
    plt.title('Distribution of pixel hit count',fontsize=14)
    plt.scatter(hit, count,s=count/10, c=hit,cmap='hot')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlim(left=0.8) 
    plt.ylim(bottom=1) 

    plt.show()


def hopalong(num, a, b, c, image_size):
    # Computes and plots the Hopalong attractor.
    points = hopalong_compute(num, a, b, c).astype(np.float32)
    hopalong_plot(points, a, b, c, num, image_size)


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


def main():
   # Main function to run the Hopalong attractor generation.
    image_size = 1000, 1000

    a = get_validated_input(
        'Enter a non-zero float value for "a": ', float, check_non_zero=True)
    b = get_validated_input('Enter a float value for "b": ', float)
    c = get_validated_input('Enter a float value for "c": ', float)
    num = get_validated_input(
        'Enter an integer value for "num": ', int)

    hopalong(num, a, b, c, image_size)


if __name__ == "__main__":
    main()
