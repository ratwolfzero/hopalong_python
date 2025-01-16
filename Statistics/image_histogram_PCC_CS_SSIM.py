import matplotlib.pyplot as plt
import numpy as np
from skimage.metrics import structural_similarity as ssim
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
    # Initialize matrices and variables
    image = np.zeros(image_size, dtype=np.uint64)
    trajectory = np.zeros((n, 2), dtype=np.float64)
    min_x, max_x, min_y, max_y = extents
    scale_x = (image_size[1] - 1) / (max_x - min_x)
    scale_y = (image_size[0] - 1) / (max_y - min_y)

    x = 0.0
    y = 0.0

    for i in range(n):
        # Store trajectory point
        trajectory[i, 0], trajectory[i, 1] = x, y

        # Map trajectory points to image pixel coordinates, rounding to nearest integer
        px = round((x - min_x) * scale_x)
        py = round((y - min_y) * scale_y)

        # Bounds check to ensure indices are within the image
        if 0 <= px < image_size[1] and 0 <= py < image_size[0]:
        # populate the image and calculate trajectory "on the fly"    
            image[py, px] += 1  # Respecting row/column convention, accumulate # of hits
        x, y = y - copysign(1.0, x) * sqrt(fabs(b * x - c)), a-x
        
    return image, trajectory
    

# Create histogram-based density matrix directly from trajectory
def create_histogram_density_matrix(trajectory, image_size):
    hist_density, x_edges, y_edges = np.histogram2d(
        trajectory[:, 0], trajectory[:, 1], bins=image_size, density=True)
    return hist_density, x_edges, y_edges
    

# Normalize matrices to a common range (0-1)
def normalize(matrix):
    min_val = np.min(matrix)
    max_val = np.max(matrix)
    return (matrix - min_val) / (max_val - min_val)

# Pearson Correlation Coefficient (PCC) function
def pearson_correlation(image, hist_density):
    return np.corrcoef(image.flatten(), hist_density.T.flatten())[0, 1]

# Cosine Similarity (CS) function
def cosine_similarity(image, hist_density):
    return np.dot(image.flatten(), hist_density.T.flatten()) / (
        np.linalg.norm(image.flatten()) * np.linalg.norm(hist_density.T.flatten())
    )

# Structural Similarity Index (SSIM) function
def structural_similarity_index(image, hist_density):
    # Normalize both matrices for SSIM
    norm_image = normalize(image)
    norm_hist_density = normalize(hist_density)
    return ssim(norm_image, norm_hist_density.T, data_range=1.0)

# Compute statistics
def compute_statistics(image, hist_density):
    # Compute similarity metrics
    ssim_value = structural_similarity_index(image, hist_density)
    pearson_corr = pearson_correlation(image, hist_density)
    cosine_sim = cosine_similarity(image, hist_density)

    return {
        "Structural Similarity Index (SSIM)": ssim_value,
        "Pearson Correlation Coefficient": pearson_corr,
        "Cosine Similarity": cosine_sim,
    }


# Plot results
def plot_density_matrices(image, hist_density, extent, x_edges, y_edges, color_map, params=None, stats=None):
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    norm_image = normalize(image)
    norm_hist_density = normalize(hist_density)

    # Pixel-Based Density Matrix
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

    # Histogram-Based Density Matrix
    title_histogram_based = 'Histogram Density Matrix'
    if params:
        title_histogram_based += f"\n(a={params['a']}, b={params['b']}, c={params['c']}, n={params['n']})"
        
    X, Y = np.meshgrid(x_edges, y_edges)
    
    im2 = axes[1].pcolormesh(X, Y, norm_hist_density.T, cmap=color_map, shading=None, norm=None, antialiased=False)
    axes[1].set_aspect('equal')  # Set equal aspect ratio explicitly for pcolormesh
    fig.colorbar(im2, ax=axes[1], label='Density')

    axes[1].set_title(title_histogram_based)
    axes[1].set_xlabel('X')
    axes[1].set_ylabel('Y')

    plt.tight_layout()
    plt.show()

   
def main(image_size=(1000, 1000), color_map='hot'):
    try:
        params = get_attractor_parameters()

        # Compute trajectory extents
        extents = compute_trajectory_extents(params['a'], params['b'], params['c'], params['n'])
        extent = [extents[0], extents[1], extents[2], extents[3]]

        # Compute image and trajectory
        image, trajectory = compute_trajectory_image(
            params['a'], params['b'], params['c'], params['n'], extents, image_size
        )

        # Create histogram-based density matrix directly from trajectory
        hist_density, x_edges, y_edges = create_histogram_density_matrix(trajectory, image_size)

        # Compute statistics
        stats = compute_statistics(image, hist_density)
        for name, value in stats.items():
            print(f"{name}: {value:.4f}")

        # Plot results
        plot_density_matrices(
            image, hist_density, extent, x_edges, y_edges, color_map, params=params, stats=stats
        )

    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == '__main__':
    main()
    