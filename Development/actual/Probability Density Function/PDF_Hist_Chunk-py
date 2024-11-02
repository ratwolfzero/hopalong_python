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
    chunk_size = validate_input('Enter a positive integer for chunk size: ', int, check_positive_non_zero=True, min_value=1)
    return {'a': a, 'b': b, 'c': c, 'num': num, 'chunk_size': chunk_size}

@njit
def compute_trajectory_chunked(a, b, c, num_iterations, start_x, start_y):
    trajectory_chunk = np.zeros((num_iterations, 2))
    x, y = start_x, start_y

    for i in range(num_iterations):
        trajectory_chunk[i, 0] = x
        trajectory_chunk[i, 1] = y
        xx = y - copysign(1.0, x) * sqrt(fabs(b * x - c))
        yy = a - x
        x = xx
        y = yy

    return trajectory_chunk, x, y

def plot_trajectory_with_density(trajectory):
    # Use 2D histogram for density estimation instead of KDE
    x = trajectory[:, 0]
    y = trajectory[:, 1]
    
    # Define the binning parameters
    bins = 500
    density, xedges, yedges = np.histogram2d(x, y, bins=bins, density=True)
    
    # Plotting with density color mapping
    plt.figure(figsize=(8, 8))
    plt.imshow(density.T, origin='lower', cmap='hot', extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])
    plt.colorbar(label='Density')
    plt.title('Hopalong Attractor with Density Coloring (2D Histogram Approximation)')
    plt.xlabel('X (Cartesian)')
    plt.ylabel('Y (Cartesian)')
    plt.axis('equal')
    plt.tight_layout()
    #plt.show()
    plt.pause(1)
    plt.close()

def main():
    try:
        params = get_attractor_parameters()
        
        total_iterations = params['num']
        chunk_size = params['chunk_size']

        trajectory = np.zeros((total_iterations, 2))
        start_time = time.process_time()
        
        x, y = np.float64(0.0), np.float64(0.0)

        for start in range(0, total_iterations, chunk_size):
            end = min(start + chunk_size, total_iterations)
            size = end - start
            
            # Compute trajectory for the current chunk
            trajectory_chunk, x, y = compute_trajectory_chunked(params['a'], params['b'], params['c'], size, x, y)
            trajectory[start:end] = trajectory_chunk

        plot_trajectory_with_density(trajectory)

        end_time = time.process_time()
        print(f'Execution time: {end_time - start_time:.2f} seconds')
        
    except Exception as e:
        print(f'An error occurred: {e}')

if __name__ == '__main__':
    main()


