import matplotlib.pyplot as plt
import numpy as np
from numba import njit
from math import copysign, sqrt, fabs
import time

def validate_input(prompt, input_type=float, check_positive_non_zero=False, min_value=None):
    while True:
        user_input = input(prompt)
        try:
            value = input_type(user_input)
            if check_positive_non_zero and value <= 0:
                print('Invalid input. The value must be a positive non-zero number.')
                continue
            if min_value is not None and value < min_value:
                print(f'Invalid input. The value should be at least {min_value}.')
                continue
            return value
        except ValueError:
            print(f'Invalid input. Please enter a valid {input_type.__name__} value.')

def get_attractor_parameters():
    a = validate_input('Enter a float value for "a": ', float)
    b = validate_input('Enter a float value for "b": ', float)
    while True:                                                 
        c = validate_input('Enter a float value for "c": ', float)
        if (a == 0 and b == 0 and c == 0) or (a == 0 and c == 0):
            print('Invalid combination of parameters. The following combinations are not allowed:\n'
                  '- a = 0, b = 0, c = 0\n'                        
                  '- a = 0, b = any value, c = 0\n'
                  'Please enter different values.')
        else:
            break
    num = validate_input('Enter a positive integer value for "num": ', int, check_positive_non_zero=True, min_value=1000)
    return {'a': a, 'b': b, 'c': c, 'num': num}

@njit
def compute_trajectory(a, b, c, num):
    trajectory = np.zeros((num, 2), dtype=np.float32)
    x, y = 0.0, 0.0

    for i in range(num):
        trajectory[i, 0] = x
        trajectory[i, 1] = y
        xx = y - copysign(1.0, x) * sqrt(fabs(b * x - c))
        yy = a - x
        x, y = xx, yy

    return trajectory

def plot_trajectory_with_histogram(trajectory, bins=100):
    # Create a 2D histogram
    hist, xedges, yedges = np.histogram2d(trajectory[:, 0], trajectory[:, 1], bins=bins, density=True)

    # Create a meshgrid for plotting
    xpos, ypos = np.meshgrid(xedges[:-1], yedges[:-1], indexing="ij")
    xpos = xpos.ravel()
    ypos = ypos.ravel()
    zpos = 0  # Starting Z position for the histogram bars

    # Flatten the histogram to get the heights
    dz = hist.ravel()

    # Create a 2D scatter plot with histogram coloring
    plt.figure(figsize=(10, 10))
    plt.imshow(hist.T, origin='lower', cmap='hot', extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], aspect='auto')
    plt.colorbar(label='Density')
    plt.title('Hopalong Attractor with Histogram-Based Density Coloring')
    plt.xlabel('X (Cartesian)')
    plt.ylabel('Y (Cartesian)')
    plt.axis('equal')
    plt.show()

def main():
    try:
        params = get_attractor_parameters()
        
        # Start the time measurement
        start_time = time.process_time()

        trajectory = compute_trajectory(params['a'], params['b'], params['c'], params['num'])
        plot_trajectory_with_histogram(trajectory, bins=100)

        # End the time measurement
        end_time = time.process_time()
        print(f'Execution time: {end_time - start_time:.2f} seconds')
        
    except Exception as e:
        print(f'An error occurred: {e}')

# Main execution
if __name__ == '__main__':
    main()
