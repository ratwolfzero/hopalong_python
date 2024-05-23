import matplotlib.pyplot as plt
import numpy as np
import matplotlib
matplotlib.use('TkAgg')

image_size = (8001, 8001)


def hopalong(num, a, b, c, image_size):
    x, y = 0, 0
    u, v = np.empty(num), np.empty(num)

    for i in range(num):
        u[i], v[i] = x, y
        xx, yy = y - np.sign(x) * np.sqrt(abs(b * x - c)), a - x
        x, y = xx, yy

    min_x, max_x = np.min(u), np.max(u)
    min_y, max_y = np.min(v), np.max(v)

    img_width, img_height = image_size
    img = np.empty((img_height, img_width))

    px = ((u - min_x) / (max_x - min_x) * (img_width-1)).astype(int)
    py = ((v - min_y) / (min_y - max_y) * (img_height-1)).astype(int)
    img[py, px] = 1

    plt.figure(figsize=(8, 8))
    plt.imshow(img, cmap='inferno')
    plt.title(f"Hopalong Attractor\nParams: a={a}, b={b}, c={c}, num={num}")
    plt.show()


print("Input the parameters a, b, c (e.g., -1.7 -0.3 0.7) and the number of iterations num (e.g., 1000000 or 1_000_000)")
a = float(input('a? '))
b = float(input('b? '))
c = float(input('c? '))
num = int(input('num? '))
hopalong(num, a, b, c, image_size)
