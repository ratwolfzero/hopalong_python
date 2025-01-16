import numpy as np
import matplotlib.pyplot as plt
from numba import njit, prange
from math import copysign, sqrt, fabs
from scipy.spatial.distance import pdist


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


# Compute trajectory image
@njit
def compute_trajectory_image(a, b, c, n, extents, image_size):
    image = np.zeros(image_size, dtype=np.uint64)
    min_x, max_x, min_y, max_y = extents
    scale_x = (image_size[1] - 1) / (max_x - min_x)
    scale_y = (image_size[0] - 1) / (max_y - min_y)
    x = 0.0
    y = 0.0

    for _ in range(n):
        px = round((x - min_x) * scale_x)
        py = round((y - min_y) * scale_y)
        if 0 <= px < image_size[1] and 0 <= py < image_size[0]:
            image[py, px] += 1
        x, y = y - copysign(1.0, x) * sqrt(fabs(b * x - c)), a-x

    return image


@njit(parallel=True)
def compute_correlation_integral(image, r):
    count = 0
    total_points = np.sum(image)
    total_pairs = total_points * (total_points - 1)

    for x in range(image.shape[0]):
        for y in range(image.shape[1]):
            if image[x, y] > 0:
                # Calculate the range only once
                for dx in range(-r, r + 1):
                    dy_limit = int(sqrt(r**2 - dx**2))  # Calculate max dy for current dx
                    for dy in range(-dy_limit, dy_limit + 1):
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < image.shape[0] and 0 <= ny < image.shape[1]:
                            count += image[x, y] * image[nx, ny]

    return count / total_pairs if total_pairs > 0 else 0


def calculate_r_values(image, min_factor=0.1, max_factor=0.5):
    
    points = np.array(np.nonzero(image)).T
    if points.size > 0:
        distances = pdist(points)
        r_min = np.min(distances) * min_factor  # Adjust with min_factor
        r_max = np.max(distances) * max_factor  # Adjust with max_factor
    else:
        r_min, r_max = 1, 100  # Fallback in case there are no points
    
    return r_min, r_max
 

# Adjust the r_values and log-log fitting
def estimate_correlation_dimension(image, r_values):
    correlations = []
    total_points = np.sum(image)
    if total_points < 2:
        raise ValueError("Insufficient points in the image to compute correlations.")
    
    for r in r_values:
        correlation = compute_correlation_integral(image, int(r))
        correlations.append(correlation)

    # Replace zero correlations with a small positive number
    correlations = np.array(correlations)
    correlations[correlations <= 0] = 1e-10

    # Perform log-log fitting for correlation dimension
    valid = correlations > 0  # Ensure only positive values are used
    log_r = np.log(r_values[valid])
    log_correlations = np.log(correlations[valid])
    slope, intercept = np.polyfit(log_r, log_correlations, 1)

    return slope, correlations


# Display results
def display_results(r_values, correlations, correlation_dimension, image, extents, params=None):
    # Plot both results in subplots
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    title_heatmap = 'Density Heatmap Matrix'
    if params:
        title_heatmap += f"\n a={params['a']}, b={params['b']}, c={params['c']}, n={params['n']}"

    # Plot attractor heatmap
    ax1 = axes[0]
    ax1.imshow(image, extent=(extents[0], extents[1], extents[2], extents[3]), origin='lower', cmap='hot')
    ax1.set_title(title_heatmap)
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')

    # Plot correlation integral
    ax2 = axes[1]
    ax2.loglog(r_values, correlations, marker='o')
    #title = 'Correlation Integral'
    title  = f"\nCorrelation Dimension:{correlation_dimension:.4f} "
    ax2.set_title(title, fontsize=12, fontweight='bold')
    ax2.set_xlabel('Radius r',fontsize=14)
    ax2.set_ylabel('Correlation Integral C(r)', fontsize=14)
    ax2.grid(True, which='both')

    # Show the combined plot
    plt.tight_layout()
    plt.show()


# Main function
def main():
    params = get_attractor_parameters()
    extents = compute_trajectory_extents(params['a'], params['b'], params['c'], params['n'])
    image_size = (1000, 1000)  
    

    image = compute_trajectory_image(params['a'], params['b'], params['c'], params['n'], extents, image_size)
    
    # Get the minimum and maximum distances from the image to set r_min and r_max using the new function
    #r_min, r_max = calculate_r_values(image)
    r_min, r_max = 7, 700
    
    # Adjust r_values based on calculated r_min and r_max
    r_values = np.logspace(np.log10(r_min), np.log10(r_max), num=20)

    # Compute correlation dimension and correlation integral
    correlation_dimension, correlations = estimate_correlation_dimension(image, r_values)

    # Plot and display results
    display_results(r_values, correlations, correlation_dimension, image, extents, params=params)

if __name__ == '__main__':
    main()
