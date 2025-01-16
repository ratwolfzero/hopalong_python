import matplotlib.pyplot as plt
import numpy as np
from skimage.metrics import structural_similarity as ssim
from numba import njit
from math import copysign, sqrt, fabs
from scipy.stats import gaussian_kde


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
def compute_trajectory_image(a, b, c, n, extents, image_size):
    image = np.zeros(image_size, dtype=np.uint64)
    trajectory = np.zeros((n, 2), dtype=np.float64)
    min_x, max_x, min_y, max_y = extents
    scale_x = (image_size[1] - 1) / (max_x - min_x)
    scale_y = (image_size[0] - 1) / (max_y - min_y)

    x = 0.0
    y = 0.0

    for i in range(n):
        trajectory[i, 0], trajectory[i, 1] = x, y
        px = round((x - min_x) * scale_x)
        py = round((y - min_y) * scale_y)
        
        if 0 <= px < image_size[1] and 0 <= py < image_size[0]:
            image[py, px] += 1  
        x, y = y - copysign(1.0, x) * sqrt(fabs(b * x - c)), a-x
        
    return image, trajectory


def compute_kde(trajectory):
    x = trajectory[:, 0]
    y = trajectory[:, 1]
    
    xy = np.vstack([x, y])
    kde = gaussian_kde(xy, bw_method=0.004)
    
    grid_size = 1000  
    xmin, xmax = np.min(x), np.max(x)
    ymin, ymax = np.min(y), np.max(y)
    xgrid = np.linspace(xmin, xmax, grid_size)
    ygrid = np.linspace(ymin, ymax, grid_size)
    X, Y = np.meshgrid(xgrid, ygrid)
    grid_coords = np.vstack([X.ravel(), Y.ravel()])
    Z = kde(grid_coords).reshape(X.shape)
    return Z, xgrid, ygrid


def normalize(matrix):
    min_val = np.min(matrix)
    max_val = np.max(matrix)
    return (matrix - min_val) / (max_val - min_val)


def pearson_correlation(image, kde):
    return np.corrcoef(image.flatten(), kde.flatten())[0, 1]


def cosine_similarity(image, kde):
    return np.dot(image.flatten(), kde.flatten()) / (
        np.linalg.norm(image.flatten()) * np.linalg.norm(kde.flatten())
    )


def structural_similarity_index(image, kde):
    norm_image = normalize(image)
    norm_KDE = normalize(kde)
    return ssim(norm_image, norm_KDE, data_range=1.0)


def compute_statistics(image, kde):
    ssim_value = structural_similarity_index(image, kde)
    pearson_corr = pearson_correlation(image, kde)
    cosine_sim = cosine_similarity(image, kde)

    return {
        "Structural Similarity Index (SSIM)": ssim_value,
        "Pearson Correlation Coefficient": pearson_corr,
        "Cosine Similarity": cosine_sim,
    }


def plot_density_matrices(image, kde, extent, xgrid, ygrid, color_map, params=None, stats=None):
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    norm_image = normalize(image)
    norm_KDE = normalize(kde)

    title_pixel_based = 'Density Heatmap Matrix'
    if stats:
        title_pixel_based += f"\n(Pearson: {stats['Pearson Correlation Coefficient']:.4f}, " \
                             f"Cosine: {stats['Cosine Similarity']:.4f}, " \
                             f"SSIM: {stats['Structural Similarity Index (SSIM)']:.4f})"

    im1 = axes[0].imshow(norm_image, origin='lower', cmap=color_map, extent=extent, interpolation='none', aspect='equal')
    axes[0].set_title(title_pixel_based)
    axes[0].set_xlabel('X')
    axes[0].set_ylabel('Y')
    fig.colorbar(im1, ax=axes[0], label='Density')

    title_kde_based = 'KDE Density Matrix'
    if params:
        title_kde_based += f"\n(a={params['a']}, b={params['b']}, c={params['c']}, n={params['n']})"
        
    X, Y = np.meshgrid(xgrid, ygrid)
    
    im2 = axes[1].pcolormesh(X, Y, norm_KDE, cmap=color_map, shading='auto', antialiased=False)
    axes[1].set_aspect('equal') 
    fig.colorbar(im2, ax=axes[1], label='Density')

    axes[1].set_title(title_kde_based)
    axes[1].set_xlabel('X')
    axes[1].set_ylabel('Y')

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

        kde, xgrid, ygrid = compute_kde(trajectory)

        stats = compute_statistics(image, kde)
        for name, value in stats.items():
            print(f"{name}: {value:.4f}")

        plot_density_matrices(
            image, kde, extent, xgrid, ygrid, color_map, params=params, stats=stats
        )

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == '__main__':
    main()