import matplotlib.pyplot as plt
import numpy as np
import matplotlib
matplotlib.use('TkAgg')


image_size = 8001, 8001


def hopalong(num, a, b, c, image_size):
    x, y = np.float64(0), np.float64(0)
    u, v = np.zeros(num, dtype=np.float64), np.zeros(num, dtype=np.float64)

    for i in range(num):
        u[i], v[i] = x, y
        xx, yy = y - np.sign(x) * np.sqrt(abs(b * x - c)), a - x
        x, y = xx, yy

    min_x, max_x = np.min(u), np.max(u)
    min_y, max_y = np.min(v), np.max(v)

    img_width, img_height = image_size
    img = np.zeros((img_height, img_width))

    px = ((u - min_x) / (max_x - min_x) * (img_width-1)).astype(np.int64)
    py = ((v - min_y) / (min_y - max_y) * (img_height-1)).astype(np.int64)

    img[py, px] = 1

    plt.figure(figsize=(8, 8))
    plt.imshow(img, cmap='inferno')
    plt.title(f"Hopalong Attractor\nParams: a={a}, b={b}, c={c}, num={num}")
    plt.show()


def get_validated_input(prompt, input_type=float, check_non_zero=False):
    while True:
        user_input = input(prompt)
        try:
            value = input_type(user_input)
        except ValueError:
            print(f"Invalid input. Please enter a valid {
                  input_type.__name__} value.")
            continue

        if check_non_zero and value == 0:
            print("Invalid input. The value cannot be zero.")
        else:
            return value


a = get_validated_input(
    'Enter a non-zero float value for "a": ', float, check_non_zero=True)
b = get_validated_input('Enter a float value for "b": ', float)
c = get_validated_input('Enter a float value for "c": ', float)
num = get_validated_input('Enter an integer value for "num": ', int)

hopalong(num, a, b, c, image_size)
