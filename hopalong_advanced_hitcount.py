import matplotlib.pyplot as plt
import numpy as np
from numba import jit

# Use TkAgg backend for matplotlib
import matplotlib
matplotlib.use('TkAgg')


@jit(nopython=True)
def compute_attractor_points(num, a, b, c):
   # Computes the points for the Hopalong attractor.
    points = np.zeros((num, 2), dtype=np.float32)
    x, y = 0.0, 0.0

    for i in range(num):

        points[i] = x, y
        xx, yy = y - np.sign(x) * np.sqrt(abs(b * x - c)), a - x
        x, y = xx, yy

    return points

def calculate_image_pixels(points, image_size, min_x, max_x, min_y, max_y):
    # calculate pixels based on points for image size
    img_width, img_height = image_size

    px = ((points[:, 0] - min_x) / (max_x - min_x) * (img_width - 1)).astype(np.int16)
    py = ((points[:, 1] - min_y) / (max_y - min_y) * (img_height - 1)).astype(np.int16)

    return px, py

@jit(nopython=True)
def image_pixels_and_hit_count(img, px, py):
    # set image pixels and track their intensity (used for color sheme cmap)
    for px_i, py_i in zip(px, py):
        img[py_i, px_i] += 1
    return img


def plot_attractor_image(img, colormap, extents, params, size=(8,8)):
    plt.figure(figsize=size)
    plt.imshow(img, cmap=colormap, origin='lower', extent=extents)
    plt.title("Hopalong Attractor @ratwolf2024\nParams: a={a}, b={b}, c={c}, num={num:_}".format(**params))
    
def plot_intensity_distribution(img, colormap, size=(10,8), scale='log'):
    hit, count = np.unique(img[img!=0], return_counts=True)
    hit_pixel = sum(j for i, j in zip(hit, count))
    img_pixel = np.prod(img.shape)
    hit_ratio = '{:02.3f}'.format(hit_pixel / img_pixel * 100)
    plt.figure(figsize=size)
    plt.xlabel('# of hits (n)',fontsize=10)
    plt.ylabel('# of pixels hit n-times',fontsize=10)
    plt.title(f'Distribution of pixel intensity. In total {hit_pixel} pixels of {img_pixel} = {hit_ratio}% have been hit')
    plt.scatter(hit, count, s=count/10, c=hit, cmap=colormap)
    plt.xscale(scale)
    plt.yscale(scale)
    plt.xlim(left=0.8) 
    plt.ylim(bottom=1) 

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


def hopalong_plot_all(points, a, b, c, num, image_size):
    # Plots the points of the Hopalong attractor.
    color_map='hot'    
        
    min_x, max_x = np.min(points[:,0]), np.max(points[:,0])
    min_y, max_y = np.min(points[:,1]), np.max(points[:,1])

    px, py = calculate_image_pixels(points, image_size, min_x, max_x, min_y, max_y)
    img = image_pixels_and_hit_count(np.zeros(image_size, dtype=np.int16), px, py)
    
    
    extents=[min_x, max_x, min_y, max_y]
    params = {'a': a, 'b': b, 'c': c, 'num': num}
    
    plot_attractor_image(img, color_map, extents, params)
    plot_intensity_distribution(img, color_map)
    
    plt.show()
    

def hopalong(num, a, b, c, image_size):
    # Computes and plots the Hopalong attractor.
    points = compute_attractor_points(num, a, b, c).astype(np.float32)
    hopalong_plot_all(points, a, b, c, num, image_size)


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
