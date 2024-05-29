import matplotlib.pyplot as plt
import numpy as np
from numba import jit
import matplotlib; matplotlib.use('TkAgg') 
#TKAgg at least required for MacOS and apple silicon chip to avoid crash of plot window during interaction

image_size = 8000, 8000


@jit(nopython=True)  #enforce just-in-time compilation to machine code
def hopalong_compute(num, a, b, c): #split hpalong in calculation and plot function to make proper use of @jit
    points = np.empty((num, 2), dtype=np.float64) #np.empty vs. np.zeros is more efficient, if all elements will be set!
    x, y = 0.0, 0.0 #python native float i.e. 0.0 is faster for scalar operation than np.float

    for i in range(num):
        points[i] = x, y #points[i] = x, y versus u[i], v[i] = x, y is more memory efficient
        xx, yy = y - np.sign(x) * np.sqrt(abs(b * x - c)), a - x
        x, y = xx, yy

    return points


def hopalong_plot(u, v, a, b, c, image_size):
    min_x, max_x = np.min(u), np.max(u)
    min_y, max_y = np.min(v), np.max(v)

    img_width, img_height = image_size
    img = np.empty((img_height, img_width))

    px = ((u - min_x) / (max_x - min_x) * (img_width - 1)).astype(np.int64)
    py = ((v - min_y) / (min_y - max_y) * (img_height - 1)).astype(np.int64)

    img[py, px] = 1

    plt.figure(figsize=(8, 8))
    plt.imshow(img, cmap='inferno')
    plt.title(f"Hopalong Attractor\nParams: a={a}, b={b}, c={c}, num={(f"{num:_}")}")
    plt.show()

#call seperated hopalong functions
def hopalong(num, a, b, c, image_size):
    points = hopalong_compute(num, a, b, c)
    hopalong_plot(points[:, 0], points[:, 1], a, b, c, image_size) 


print("Input the parameters a, b, c as float (e.g., -1.7 -0.3 0.7) and the number of iterations num as int (e.g., 1000000 or 1_000_000)")
#use maximum 100_000_000 iterations to avoid memory overflow respectively swap! (8 GByte RAM)

def get_input(prompt, input_type=float, check_non_zero=False):
    while True:
        try:
            value = input_type(input(prompt))
            if check_non_zero and value == 0:
                print("Invalid input. The value cannot be zero.")
            else:
                return value
        except ValueError:
            print(f"Invalid input. Please enter a valid {input_type.__name__} number.")

a = get_input('a? ', float, check_non_zero=True)
b = get_input('b? ', float)
c = get_input('c? ', float)
num = get_input('num? ', int)

hopalong(num, a, b, c, image_size)
