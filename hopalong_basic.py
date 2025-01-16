import matplotlib.pyplot as plt
import numpy as np
from numba import njit
from math import copysign, sqrt, fabs
import time
import resource 


def validate_input(prompt, input_type=float, check_positive_non_zero=False, min_value=None):
    # Prompt for and return user input validated by type and specific checks.
    while True:
        try:
            value = float(input(prompt))
            if input_type == int:
                if not value.is_integer():
                    raise ValueError('Please enter an integer.')
                value = int(value)
            if check_positive_non_zero and value <= 0:
                raise ValueError('The value must be positive and non-zero.')
            if min_value is not None and value < min_value:
                raise ValueError(f'The value must be at least {min_value}.')
            return value
        except ValueError as e:
            print(f'Invalid input. Please enter a valid {input_type.__name__} value. ({e})')
            
            
def validate_attractor_parameters(a, b, c):
    while a == 0 and c == 0:
        print('Invalid combination of parameters: a=0, b=0, c=0 or a=0, b=any, c=0')
        c = validate_input('Enter a float value for "c": ')
    return a, b, c
            

def get_attractor_parameters():
    a = validate_input('Enter a float value for "a": ')
    b = validate_input('Enter a float value for "b": ')
    c = validate_input('Enter a float value for "c": ')
    a, b, c = validate_attractor_parameters(a, b, c)
    n = validate_input('Enter a positive integer value > 1000 for "n": ', int, True, 1000)
    return {'a': a, 'b': b, 'c': c, 'n': n}


@njit #njit is an alias for nopython=True
def compute_trajectory_extents(a, b, c, n):
    # Dynamically compute and track the minimum and maximum extents of the trajectory over 'n' iterations.
    x = 0.0
    y = 0.0

    min_x = float('inf')  # ensure that the initial minimum is determined correctly
    max_x = float('-inf') # ensure that the initial maximum is determined correctly
    min_y = float('inf')
    max_y = float('-inf')

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
        x, y = y - copysign(1.0, x) * sqrt(fabs(b * x - c)), a-x
           
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
    
    x = 0.0
    y = 0.0
    
    for _ in range(n):
        # Map trajectory points to image pixel coordinates, rounding to nearest integer
        px = round((x - min_x) * scale_x)
        py = round((y - min_y) * scale_y)

        # Bounds check to ensure indices are within the image
        if 0 <= px < image_size[1] and 0 <= py < image_size[0]:
        # populate the image and calculate trajectory "on the fly"    
            image[py, px] += 1  # Respecting row/column convention, accumulate # of hits
        x, y = y - copysign(1.0, x) * sqrt(fabs(b * x - c)), a-x
        
    return image

# Dummy call to ensure the function is pre-compiled by the JIT compiler before it's called by the interpreter.
_ = compute_trajectory_and_image(1.0, 1.0, 1.0, 2, (-1, 0, 0, 1), (2, 2))


# Plot Setup
def setup_plot(ax, title=None, xlabel=None, ylabel=None):
    if title:
        ax.set_title(title)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)


# Create colorbar
def colorbar(image, fig, ax, img):
    cbar = fig.colorbar(img, ax=ax, location='bottom')
    cbar.set_label('Pixel Density. (Scale = 0 - max)')
    max_hit_count = np.max(image)
    tick_positions = np.linspace(0, max_hit_count, num = 8)  # Choose 8 tick positions
    tick_labels = (int(tick) for tick in tick_positions)  # Format tick labels as integers
    cbar.set_ticks(tick_positions)  # Set ticks on the colorbar
    cbar.set_ticklabels(tick_labels)  # Set formatted labels
      
    
# Render 2D
def render_trajectory_image(image, extents, params, color_map):
    fig = plt.figure(figsize=(8, 8), facecolor='gainsboro')
    ax = fig.add_subplot(111)
    img = ax.imshow(image, origin='lower', cmap=color_map, extent=extents, interpolation='none')

    title = f'Hopalong Attractor@ratwolf@2024\nParams: a={params["a"]}, b={params["b"]}, c={params["c"]}, n={params["n"]:_}'
    setup_plot(ax, title, 'X (Cartesian)', 'Y (Cartesian)')

    colorbar(image, fig, ax, img)

    plt.tight_layout()
    plt.show()
    #plt.pause(1)
    #plt.close(fig)


def calculate_and_display_resource_usage(start_time, end_time):
    # Calculate the CPU user and system time
    cpu_sys_time_used = end_time - start_time

    # Calculate the memory resources used
    memMb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0 / 1024.0

    print(f'CPU User&System time: {cpu_sys_time_used:.2f} seconds')
    print(f'Memory (RAM): {memMb:.2f} MByte used')


def main(image_size=(1000, 1000), color_map='hot'):
    # Main execution process
    try:
        params = get_attractor_parameters()
        
        # Start the time measurement
        start_time = time.process_time()

        extents = compute_trajectory_extents(params['a'], params['b'], params['c'], params['n'])
        image = compute_trajectory_and_image(params['a'], params['b'], params['c'], params['n'], extents, image_size)
        render_trajectory_image(image, extents, params, color_map)

        # End the time measurement
        end_time = time.process_time()

        calculate_and_display_resource_usage(start_time, end_time)
        
    except Exception as e:
        print(f'An error occurred: {e}')


# Main execution
if __name__ == '__main__':
    main()