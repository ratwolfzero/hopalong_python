import matplotlib.pyplot as plt
import numpy as np
from numba import njit
from math import copysign, sqrt, fabs
import time
import resource

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
    x = np.float64(0.0)
    y = np.float64(0.0)      
    trajectory = np.zeros((num, 2))

    for i in range(num):
        trajectory[i, 0] = x
        trajectory[i, 1] = y
        xx = y - copysign(1.0, x) * sqrt(fabs(b * x - c))
        yy = a - x
        x = xx
        y = yy

    return trajectory


def plot_trajectory_with_density(trajectory):
    # Use 2D histogram for density estimation
    x = trajectory[:, 0]
    y = trajectory[:, 1]
    
    # Define the binning parameters
    bins = 1000
    density, xedges, yedges = np.histogram2d(x, y, bins=bins, density=True)
    
    # Create a figure
    fig = plt.figure(figsize=(8, 8),facecolor='gainsboro')
    ax = fig.add_subplot(1, 1, 1)
    
    # Plotting with density color mapping
    img = ax.imshow(density.T, origin='lower', cmap='hot', extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])
    
    # Add color bar using the ax argument
    cbar = fig.colorbar(img, ax=ax, location='bottom')
    cbar.set_label('Density')  # Title for colorbar

    ax.set_title('Hopalong Attractor with Density Coloring (2D Histogram Approximation)')
    ax.set_xlabel('X (Cartesian)')
    ax.set_ylabel('Y (Cartesian)')
    #ax.axis('equal')
    plt.tight_layout()
    plt.show()


def main():
    try:
        params = get_attractor_parameters()
        
        # Start the time measurement
        start_time = time.process_time()

        trajectory = compute_trajectory(params['a'], params['b'], params['c'], params['num'])
        plot_trajectory_with_density(trajectory)

        # End the time measurement
        end_time = time.process_time()

        # Calculate the CPU user and system time
        cpu_sys_time_used = end_time - start_time

        # Calculate the memory resources used
        memMb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024.0/1024.0
        
        print(f'CPU User&System time used: {cpu_sys_time_used:.2f} seconds')
        print (f'Memory (RAM): {memMb:.2f} MByte used')
        
    except Exception as e:
        print(f'An error occurred: {e}')


if __name__ == '__main__':
    main()







