import numpy as np
import matplotlib
matplotlib.use('TkAg') # without backend definition works fine with MacOS. However crashes when trying to manually resize the created plot window.
import matplotlib.pyplot as plt


def hopalong(num, a, b, c, image_size=(1001, 1001)):
    x, y = 0, 0
    u, v = [], []

    min_x, max_x = float('inf'), float('-inf')
    min_y, max_y = float('inf'), float('-inf')

    for i in range(num):
        u.append(x)
        v.append(y)
        xx = y - np.sign(x) * np.sqrt(abs(b * x - c))
        yy = a - x
        x = xx
        y = yy

        min_x = min(min_x, x)
        max_x = max(max_x, x)
        min_y = min(min_y, y)
        max_y = max(max_y, y)

    img_width, img_height = image_size
    img = np.zeros((img_height, img_width))

    for i in range(num - 1):
        px = int((u[i] - min_x) / (max_x - min_x) * (img_width - 1))
        py = int((v[i] - min_y) / (min_y - max_y) * (img_height - 1))
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
hopalong(num, a, b, c)
