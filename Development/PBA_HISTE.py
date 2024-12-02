import matplotlib.pyplot as plt
import numpy as np
from numba import njit
from math import copysign, sqrt, fabs
from scipy.stats import pearsonr


def validate_input(prompt, input_type=float, check_positive_non_zero=False, min_value=None):
    while True:
        user_input = input(prompt)
        try:
            value = float(user_input)
            if input_type == int and not value.is_integer():
                print('Invalid input. Please enter an integer.')
                continue
            value = int(value) if input_type == int else value
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
            print('Invalid combination of parameters. Please re-enter.')
        else:
            break
    n = validate_input('Enter a positive integer value > 1000 for "n": ', int, check_positive_non_zero=True, min_value=1000)
    return {'a': a, 'b': b, 'c': c, 'n': n}


@njit
def compute_trajectory_extents(a, b, c, n):
    x, y = 0.0, 0.0
    min_x, max_x, min_y, max_y = np.inf, -np.inf, np.inf, -np.inf
    for _ in range(n):
        min_x, max_x = min(min_x, x), max(max_x, x)
        min_y, max_y = min(min_y, y), max(max_y, y)
        xx = y - copysign(1.0, x) * sqrt(fabs(b * x - c))
        yy = a - x
        x, y = xx, yy
    return min_x, max_x, min_y, max_y


@njit
def compute_trajectory_and_image(a, b, c, n, extents, image_size):
    image = np.zeros(image_size, dtype=np.uint64)
    min_x, max_x, min_y, max_y = extents
    scale_x = (image_size[1] - 1) / (max_x - min_x)
    scale_y = (image_size[0] - 1) / (max_y - min_y)
    x, y = 0.0, 0.0
    for _ in range(n):
        px = int((x - min_x) * scale_x)
        py = int((y - min_y) * scale_y)
        image[py, px] += 1
        xx = y - copysign(1.0, x) * sqrt(fabs(b * x - c))
        yy = a - x
        x, y = xx, yy
    return image


@njit
def compute_trajectory(a, b, c, num):
    x, y = np.float64(0.0), np.float64(0.0)
    trajectory = np.zeros((num, 2))
    for i in range(num):
        trajectory[i, 0], trajectory[i, 1] = x, y
        xx = y - copysign(1.0, x) * sqrt(fabs(b * x - c))
        yy = a - x
        x, y = xx, yy
    return trajectory


def compute_statistics(image, hist_density):
    """
    Computes statistical measures between the pixel-based and histogram-based density matrices.

    Parameters:
        image (ndarray): The pixel-based density matrix.
        hist_density (ndarray): The histogram-based density matrix.

    Returns:
        dict: A dictionary containing the Pearson correlation and cosine similarity.
    """
    # Flatten matrices for comparison
    image_flat = image.flatten()
    hist_density_flat = hist_density.T.flatten()  # Transpose histogram matrix for alignment
    
    # Normalize the flattened matrices
    image_flat = (image_flat - np.min(image_flat)) / (np.max(image_flat) - np.min(image_flat))
    hist_density_flat = (hist_density_flat - np.min(hist_density_flat)) / (np.max(hist_density_flat) - np.min(hist_density_flat))

    # Pearson Correlation Coefficient
    pearson_corr = np.corrcoef(image_flat, hist_density_flat)[0, 1]

    # Cosine Similarity
    cosine_sim = np.dot(image_flat, hist_density_flat) / (
        np.linalg.norm(image_flat) * np.linalg.norm(hist_density_flat)
    )

    return {
        "Pearson Correlation Coefficient": pearson_corr,
        "Cosine Similarity": cosine_sim
    }


def plot_density_matrices(image, hist_density, extent, color_map='hot'):
    """
    Plots the pixel-based and histogram-based density matrices.

    Parameters:
        image (ndarray): The pixel-based density matrix.
        hist_density (ndarray): The histogram-based density matrix.
        extent (list): Extents for the plot axes [min_x, max_x, min_y, max_y].
        color_map (str): The colormap for the plots.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # Pixel-Based Density Matrix
    im1 = axes[0].imshow(image, origin='lower', cmap=color_map, extent=extent, interpolation='none')
    axes[0].set_title('Pixel-Based Density Matrix')
    axes[0].set_xlabel('X')
    axes[0].set_ylabel('Y')
    fig.colorbar(im1, ax=axes[0], label='Density')

    # Histogram-Based Density Matrix
    im2 = axes[1].imshow(hist_density.T, origin='lower', cmap=color_map, extent=extent, interpolation='none')
    axes[1].set_title('Histogram-Based Density Matrix')
    axes[1].set_xlabel('X')
    axes[1].set_ylabel('Y')
    fig.colorbar(im2, ax=axes[1], label='Density')

    plt.tight_layout()
    plt.show()


def main(image_size=(1000, 1000), color_map='hot'):
    try:
        # Step 1: Get attractor parameters
        params = get_attractor_parameters()

        # Step 2: Compute trajectory extents
        extents = compute_trajectory_extents(params['a'], params['b'], params['c'], params['n'])
        min_x, max_x, min_y, max_y = extents

        # Step 3: Compute pixel-based density matrix
        image = compute_trajectory_and_image(params['a'], params['b'], params['c'], params['n'], extents, image_size)

        # Step 4: Compute trajectory points
        trajectory = compute_trajectory(params['a'], params['b'], params['c'], params['n'])

        # Step 5: Compute histogram-based density matrix using np.histogram2d
        hist_density, x_edges, y_edges = np.histogram2d(
            trajectory[:, 0], trajectory[:, 1], bins=image_size, density=True
        )

        # Step 6: Compute and print statistics
        stats = compute_statistics(image, hist_density)
        for name, value in stats.items():
            print(f"{name}: {value:.4f}")

        # Step 7: Plot density matrices
        plot_density_matrices(image, hist_density, [min_x, max_x, min_y, max_y], color_map=color_map)

    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == '__main__':
    main()