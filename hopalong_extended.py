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


def calculate_hit_metrics(img):
    hit, count = np.unique(img[img > 0], return_counts=True)

    if len(hit) == 0:
        return {
            'hit': np.array([]),
            'count': np.array([]),
            'hit_for_max_count': None,
            'count_for_max_hit': None,
            'hit_pixel': 0,
            'img_points': img.size,
            'hit_ratio': 0.0,
        }

    max_count_index = np.argmax(count)
    hit_for_max_count = hit[max_count_index]
    max_hit_index = np.argmax(hit)
    count_for_max_hit = count[max_hit_index]

    hit_pixel = count.sum()
    img_pixels = img.size
    hit_ratio = hit_pixel / img_pixels * 100

    hit_metrics = {
        'hit': hit,
        'count': count,
        'hit_for_max_count': hit_for_max_count,
        'count_for_max_hit': count_for_max_hit,
        'hit_pixel': hit_pixel,
        'img_points': img_pixels,
        'hit_ratio': round(hit_ratio, 2),
    }
    return hit_metrics


def setup_plot(ax, title=None, xlabel=None, ylabel=None):
    if title:
        ax.set_title(title)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)


def render_trajectory_image(ax, img, extents, params, color_map):
    ax.imshow(img, origin='lower', cmap=color_map, extent=extents, interpolation='none')

    title = 'Hopalong Attractor@ratwolf@2024\nParams: a={a}, b={b}, c={c}, n={n:_}'.format(**params)
    setup_plot(ax, title=title, xlabel='X (Cartesian)', ylabel='Y (Cartesian)')


def plot_hit_metrics(ax, hit_metrics, scale='log'):
    ax.plot(
        hit_metrics['hit'], hit_metrics['count'],
        'o-', color='navy', markersize=1, linewidth=0.6
    )
    
    title_text = (
        f'Distribution of pixel hit count.\n'
        f'{hit_metrics["hit_pixel"]} pixels out of {hit_metrics["img_points"]} image pixels = {hit_metrics["hit_ratio"]}% have been hit at least one time.\n'
        f'The highest number of pixels with the same number of hits is {np.max(hit_metrics["count"])} with {hit_metrics["hit_for_max_count"]} hits.\n'
        f'The highest number of hits is {np.max(hit_metrics["hit"])} with {hit_metrics["count_for_max_hit"]} pixels hit.'
    )
    
    setup_plot(ax, title=title_text, xlabel='# of hits (n)', ylabel='# of pixels hit n-times')
    
    ax.set_title(title_text, fontsize=10)
    ax.set_xscale(scale)
    ax.set_yscale(scale)
    ax.set_ylim(bottom=0.9)
    ax.set_facecolor('lightgrey')  
    ax.grid(True, which='both')  


def visualize_trajectory_image_and_hit_metrics(img, extents, params, color_map, hit_metrics):
    fig = plt.figure(figsize=(18, 8),facecolor='gainsboro')

    ax1 = fig.add_subplot(1, 2, 1, aspect='auto')
    render_trajectory_image(ax1, img, extents, params, color_map)

    ax2 = fig.add_subplot(1, 2, 2, aspect='auto')
    plot_hit_metrics(ax2, hit_metrics)
    #plt.savefig('hopalong.svg', format='svg', dpi=1200)
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
        hit_metrics = calculate_hit_metrics(image)
        visualize_trajectory_image_and_hit_metrics(image, extents, params, color_map, hit_metrics)

        # End the time measurement
        end_time = time.process_time()
        
        calculate_and_display_resource_usage(start_time, end_time)

    except Exception as e:
        print(f'An error occurred: {e}')


# Main execution
if __name__ == '__main__':
    main()