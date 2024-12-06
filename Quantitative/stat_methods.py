import matplotlib.pyplot as plt
import numpy as np
from numba import njit
from math import copysign, sqrt, fabs
from scipy.spatial.distance import jensenshannon
from skimage.metrics import structural_similarity
from scipy.stats import wasserstein_distance


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
    x, y = np.float64(0.0), np.float64(0.0)
    min_x, max_x, min_y, max_y = np.inf, -np.inf, np.inf, -np.inf
    for _ in range(n):
        min_x, max_x = min(min_x, x), max(max_x, x)
        min_y, max_y = min(min_y, y), max(max_y, y)
        xx = y - copysign(1.0, x) * sqrt(fabs(b * x - c))
        yy = a - x
        x, y = xx, yy
    return min_x, max_x, min_y, max_y


@njit
def compute_trajectory_image(a, b, c, n, extents, image_size):
    image = np.zeros(image_size, dtype=np.uint64)
    trajectory = np.zeros((n, 2), dtype=np.float64)
    min_x, max_x, min_y, max_y = extents
    scale_x = (image_size[1] - 1) / (max_x - min_x)
    scale_y = (image_size[0] - 1) / (max_y - min_y)

    x, y = np.float64(0.0), np.float64(0.0)

    for i in range(n):
        trajectory[i, 0], trajectory[i, 1] = x, y
        px = int((x - min_x) * scale_x)
        py = int((y - min_y) * scale_y)
        image[py, px] += 1
        xx = y - copysign(1.0, x) * sqrt(fabs(b * x - c))
        yy = a - x
        x, y = xx, yy

    return image, trajectory


# New statistical functions
def mean_absolute_error(image, hist_density):
    return np.mean(np.abs(image.flatten() - hist_density.T.flatten()))


def root_mean_square_error(image, hist_density):
    return np.sqrt(np.mean((image.flatten() - hist_density.T.flatten())**2))


def jensen_shannon_divergence(image, hist_density):
    image_norm = image.flatten() / np.sum(image)
    hist_norm = hist_density.T.flatten() / np.sum(hist_density)
    return jensenshannon(image_norm, hist_norm)**2


def structural_similarity_index(image, hist_density):
    return structural_similarity(image, hist_density.T, data_range=image.max() - image.min())


def intersection_over_union(image, hist_density):
    image_norm = image.flatten() / np.sum(image)
    hist_norm = hist_density.T.flatten() / np.sum(hist_density)
    return np.sum(np.minimum(image_norm, hist_norm)) / np.sum(np.maximum(image_norm, hist_norm))


def earth_movers_distance(image, hist_density):
    return wasserstein_distance(image.flatten(), hist_density.T.flatten())


# Updated compute_statistics
def compute_statistics(image, hist_density):
    pearson_corr = np.corrcoef(image.flatten(), hist_density.T.flatten())[0, 1]
    cosine_sim = np.dot(image.flatten(), hist_density.T.flatten()) / (
        np.linalg.norm(image.flatten()) * np.linalg.norm(hist_density.T.flatten())
    )
    mae = mean_absolute_error(image, hist_density)
    rmse = root_mean_square_error(image, hist_density)
    jsd = jensen_shannon_divergence(image, hist_density)
    ssim = structural_similarity_index(image, hist_density)
    iou = intersection_over_union(image, hist_density)
    emd = earth_movers_distance(image, hist_density)

    return {
        "Pearson Correlation Coefficient": pearson_corr,
        "Cosine Similarity": cosine_sim,
        "Mean Absolute Error": mae,
        "Root Mean Square Error": rmse,
        "Jensen-Shannon Divergence": jsd,
        "Structural Similarity Index": ssim,
        "Intersection Over Union": iou,
        "Earth Mover's Distance": emd,
    }


def plot_density_matrices(image, hist_density, extent, x_edges, y_edges, color_map, params=None, stats=None):
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))

    title_pixel_based = 'Density Heatmap Matrix'
    if stats:
        title_pixel_based += f"\nPearson: {stats['Pearson Correlation Coefficient']:.4f}, " \
                             f"Cosine: {stats['Cosine Similarity']:.4f}"
        
    image = image / np.max(image)
    im1 = axes[0].imshow(image, origin='lower', cmap=color_map, extent=extent, interpolation='none', aspect='equal')
    axes[0].set_title(title_pixel_based)
    axes[0].set_xlabel('X')
    axes[0].set_ylabel('Y')
    fig.colorbar(im1, ax=axes[0], label='Density')

    hist_density = hist_density / np.max(hist_density)
    title_histogram_based = 'Histogram Density Matrix'
    if params:
        title_histogram_based += f"\n(a={params['a']}, b={params['b']}, c={params['c']}, n={params['n']})"
    X, Y = np.meshgrid(x_edges, y_edges)
    im2 = axes[1].pcolormesh(X, Y, hist_density.T, cmap=color_map, shading='auto')
    axes[1].set_title(title_histogram_based)
    axes[1].set_xlabel('X')
    axes[1].set_ylabel('Y')
    axes[1].set_aspect('equal')  # Set equal aspect ratio explicitly for pcolormesh
    fig.colorbar(im2, ax=axes[1], label='Density')

    plt.tight_layout()
    plt.show()


def main(image_size=(1000, 1000), color_map='hot'):
    try:
        params = get_attractor_parameters()
        extents = compute_trajectory_extents(params['a'], params['b'], params['c'], params['n'])
        extent = [extents[0], extents[1], extents[2], extents[3]]

        image, trajectory = compute_trajectory_image(
            params['a'], params['b'], params['c'], params['n'], extents, image_size
        )

        hist_density, x_edges, y_edges = np.histogram2d(
            trajectory[:, 0], trajectory[:, 1], bins=image_size, density=True
        )

        stats = compute_statistics(image, hist_density)
        for name, value in stats.items():
            print(f"{name}: {value:.4f}")

        plot_density_matrices(image, hist_density, extent, x_edges, y_edges, color_map, params=params, stats=stats)

    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == '__main__':
    main()
