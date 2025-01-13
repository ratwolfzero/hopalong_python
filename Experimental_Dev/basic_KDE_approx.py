import matplotlib.pyplot as plt
import numpy as np
from numba import njit
from math import copysign, sqrt, fabs
from scipy.ndimage import gaussian_filter
import time
import resource 


def validate_input(prompt, input_type=float, check_positive_non_zero=False, min_value=None):
    # Prompt for and return user input validated by type and positive/non-zero checks.
    while True:
        user_input = input(prompt)
        try:
            # Parse input as float first to handle scientific notation
            value = float(user_input)
            
            # Ensure the input is an integer, if expected
            if input_type == int:
                if not value.is_integer():
                    print('Invalid input. Please enter an integer.')
                    continue
                value = int(value)

            # Check if input is a positive non-zero value
            if check_positive_non_zero and value <= 0:
                print('Invalid input. The value must be a positive non-zero number.')
                continue

            # Then, check minimum value
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
    n = validate_input('Enter a positive integer value > 1000 for "n": ', int, check_positive_non_zero=True, min_value=1000)
    
    return {'a': a, 'b': b, 'c': c, 'n': n}


@njit #njit is an alias for nopython=True
def compute_trajectory_extents(a, b, c, n):
    # Dynamically compute and track the minimum and maximum extents of the trajectory over 'n' iterations.
    x = np.float64(0.0)
    y = np.float64(0.0)

    min_x = np.inf  # ensure that the initial minimum is determined correctly
    max_x = -np.inf # ensure that the initial maximum is determined correctly
    min_y = np.inf
    max_y = -np.inf

    for _ in range(n):
    # selective min/max update using direct comparisons avoiding min/max function
        if x < min_x:
            min_x = x
        if x > max_x:
            max_x = x
        if y < min_y:
            min_y = y
        if y > max_y:
            max_y = y
        # signum function respecting the behavior of floating point numbers according to IEEE 754 (signed zero)
        xx = y - copysign(1.0, x) * sqrt(fabs(b * x - c))
        yy = a-x
        x = xx
        y = yy
        
    return min_x, max_x, min_y, max_y

# Dummy call to ensure the function is pre-compiled by the JIT compiler before it's called by the interpreter.
_ = compute_trajectory_extents(1.0, 1.0, 1.0, 2)


@njit
def compute_trajectory_and_image(a, b, c, n, extents, image_size):
    # Compute the trajectory and populate the image with trajectory points
    image = np.zeros(image_size, dtype=np.uint64)
    
    # pre-compute image scale factors
    min_x, max_x, min_y, max_y = extents
    scale_x = (image_size[1] - 1) / (max_x - min_x) # column
    scale_y = (image_size[0] - 1) / (max_y - min_y) # row
    
    x = np.float64(0.0)
    y = np.float64(0.0)
    
    for _ in range(n):
        # Map trajectory points to image pixel coordinates, rounding to nearest integer
        px = round((x - min_x) * scale_x)
        py = round((y - min_y) * scale_y)

        # Bounds check to ensure indices are within the image
        if 0 <= px < image_size[1] and 0 <= py < image_size[0]:
        # populate the image and calculate trajectory "on the fly"    
            image[py, px] += 1  # Respecting row/column convention, accumulate # of hits
        xx = y - copysign(1.0, x) * sqrt(fabs(b * x - c))
        yy = a-x
        x = xx
        y = yy
        
    return image

# Dummy call to ensure the function is pre-compiled by the JIT compiler before it's called by the interpreter.
_ = compute_trajectory_and_image(1.0, 1.0, 1.0, 2, (-1, 0, 0, 1), (2, 2))


def smooth_image(image, sigma=None):
    
    return gaussian_filter(image, sigma=sigma)


def render_trajectory_image(image, extents, params, color_map):
    # Render the trajectory image in 2D
    fig = plt.figure(figsize=(8, 8),facecolor='gainsboro')
    ax = fig.add_subplot(1, 1, 1)
    
    # Display the image
    img = ax.imshow(image, origin='lower', cmap=color_map, extent=extents, interpolation='none')

    ax.set_title('Hopalong Attractor@ratwolf@2024\nParams: a={a}, b={b}, c={c}, n={n:_}'.format(**params))
    ax.set_xlabel('X (Cartesian)')
    ax.set_ylabel('Y (Cartesian)')

    #plt.savefig('hopalong.svg', format='svg', dpi=1200)

    # Create the colorbar
    cbar = fig.colorbar(img, ax=ax, location='bottom')
    cbar.set_label('Pixel Density. (Scale = 0 - max)')  # Title for colorbar

    # Set ticks to display the exact max hit count
    max_hit_count = np.max(image)  # Get the maximum hit count from the image
    tick_positions = np.linspace(0, max_hit_count, num = 8)  # Choose 8 tick positions
    tick_labels = (int(tick) for tick in tick_positions)  # Format tick labels as integers

    cbar.set_ticks(tick_positions)  # Set ticks on the colorbar
    cbar.set_ticklabels(tick_labels)  # Set formatted labels

    #ax.axis('equal')
    plt.tight_layout()
    plt.show()
    #plt.pause(1)
    #plt.close(fig)

"""
def render_trajectory_image(image, extents, params, color_map):
    # Render the trajectory image in 3D
    # Create a meshgrid for X and Y coordinates                    
    x = np.linspace(extents[0], extents[1], image.shape[1])
    y = np.linspace(extents[2], extents[3], image.shape[0])						
    x, y = np.meshgrid(x, y)

    # Plot with normalized density (hit count) as Z values
    z = image / np.max(image) if np.max(image) > 0 else image

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.contourf3D(x, y, z, levels=100, cmap=color_map)

    # Customize the plot
    ax.set_title(f'Hopalong Attractor - 3D Density (Z) Plot\nParams: a={params["a"]}, b={params["b"]}, c={params["c"]}, n={params["n"]:_}')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.view_init(elev=75, azim=-95)  # Adjust angle for better view

    plt.show()
"""

def main(image_size=(1000, 1000), color_map='hot'):
    # Main execution process
    try:
        params = get_attractor_parameters()
        
        # Start the time measurement
        start_time = time.process_time()

        extents = compute_trajectory_extents(params['a'], params['b'], params['c'], params['n'])
        image = compute_trajectory_and_image(params['a'], params['b'], params['c'], params['n'], extents, image_size)
        smoothed_image = smooth_image(image, sigma=0.75)
        render_trajectory_image(smoothed_image, extents, params, color_map)

        # End the time measurement
        end_time = time.process_time()

        # Calculate the CPU user and system time
        cpu_sys_time_used = end_time - start_time

        # Calculate the memory resources used
        memMb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024.0/1024.0
        
        print(f'CPU User&System time: {cpu_sys_time_used:.2f} seconds')
        print (f'Memory (RAM): {memMb:.2f} MByte used')
        
    except Exception as e:
        print(f'An error occurred: {e}')


# Main execution
if __name__ == '__main__':
    main()


