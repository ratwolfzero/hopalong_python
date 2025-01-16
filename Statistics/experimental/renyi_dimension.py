import numpy as np
import matplotlib.pyplot as plt
from numba import njit
from math import copysign, sqrt, fabs


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


# Compute image and trajectory
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


# Normalize the density matrix
def normalize(matrix):
    min_val = np.min(matrix)
    max_val = np.max(matrix)
    return (matrix - min_val) / (max_val - min_val)


# Compute Rényi dimension
def compute_renyi_dimension(image, q_values):
    results = {}
    normalized_image = normalize(image)
    total_pixels = np.sum(normalized_image)
    probabilities = normalized_image / total_pixels

    for q in q_values:
        if q == 1:
            # Special case for q=1 (Shannon entropy)
            entropy = -np.sum(probabilities * np.log(probabilities + 1e-10))
            renyi_dimension = entropy
        else:
            renyi_sum = np.sum(probabilities**q)
            renyi_dimension = (1 / (1 - q)) * np.log(renyi_sum)
        results[q] = renyi_dimension
    return results


# Plot results
def plot_density_and_renyi(image, renyi_results, q_values, params, extents):
    fig, ax = plt.subplots(1, 2, figsize=(16, 8))

    # Plot density matrix
    min_x, max_x, min_y, max_y = extents
    ax[0].imshow(image, origin='lower', cmap='hot', extent=(min_x, max_x, min_y, max_y), aspect='auto')
    ax[0].set_title(f"Density Heatmap Matrix\n a={params['a']}, b={params['b']}, c={params['c']}, n={params['n']}")
    ax[0].set_xlabel("X")
    ax[0].set_ylabel("Y")

    # Plot Rényi dimension
    ax[1].plot(q_values, list(renyi_results.values()), marker='o', linestyle='--')
    ax[1].set_title("Rényi Dimension")
    ax[1].set_xlabel("q")
    ax[1].set_ylabel("D_q")
    ax[1].grid(True)

    plt.tight_layout()
    plt.show()


# Main function
def main(image_size=(500, 500), q_values=None):
    if q_values is None:
        q_values = np.linspace(0.5, 3.0, 10)

    params = get_attractor_parameters()
    extents = compute_trajectory_extents(params['a'], params['b'], params['c'], params['n'])
    density_matrix = compute_trajectory_image(params['a'], params['b'], params['c'], params['n'], extents, image_size)
    renyi_results = compute_renyi_dimension(density_matrix, q_values)

    plot_density_and_renyi(density_matrix, renyi_results, q_values, params, extents)

if __name__ == '__main__':
    main()
