import matplotlib.pyplot as plt
import numpy as np
from numba import njit
from math import copysign, sqrt, fabs
import time
import resource


def get_validated_input(prompt, input_type=float, check_non_zero=False, check_positive=False):
    # Prompt for and return user input validated by type and positive/non-zero checks
    while True:
        user_input = input(prompt)
        try:
            value = input_type(user_input)
            if check_non_zero and value == 0:
                print("Invalid input. The value cannot be zero.")
                continue
            if check_positive and value <= 0:
                print("Invalid input. The value must be a positive number.")
                continue
            return value
        except ValueError:
            print(f"Invalid input. Please enter a valid {input_type.__name__} value.")


def get_attractor_parameters():
    # Prompt user to input parameters for the Hopalong Attractor
    params = {
        'a': get_validated_input('Enter a float value for "a": ', float),
        'b': get_validated_input('Enter a float value for "b": ', float),
        'c': get_validated_input('Enter a float value for "c": ', float),
        'num': get_validated_input('Enter a positive integer value for "num": ', int, check_non_zero=True, check_positive=True)
    }
    return params


@njit
def compute_full_trajectory_extents(a, b, c, num):
    # Compute the x and y extents of the Hopalong attractor trajectory.
    x = y = np.float64(0)
    min_x = min_y = np.inf
    max_x = max_y = -np.inf
    for _ in range(num):
        min_x = min(min_x, x); max_x = max(max_x, x)
        min_y = min(min_y, y); max_y = max(max_y, y)
        xx, yy = y - copysign(1.0, x) * sqrt(fabs(b * x - c)), a - x
        x, y = xx, yy
    return min_x, max_x, min_y, max_y


def generate_chunk_sizes(num, chunk_size):
    # Generator function to yield chunk sizes to process in each iteration until covering the entire range.
    for i in range(0, num, chunk_size):
        current_chunk_size = min(chunk_size, num - i)
        yield current_chunk_size


@njit
def compute_trajectory_chunk(a, b, c, current_chunk_size, x0, y0):
    # Compute a chunk of the Hopalong trajectory.
    points = np.zeros((current_chunk_size, 2), dtype=np.float64)
    x, y = x0, y0
    for i in range(current_chunk_size):
        points[i] = x, y
        xx, yy = y - copysign(1.0, x) * sqrt(fabs(b * x - c)), a - x
        x, y = xx, yy
    return points, x, y


def compute_pdf_histogram(a, b, c, num, chunk_size, extents, bins):
    # Compute the histogram for the full trajectory, accumulating PDF data in chunks.
    min_x, max_x, min_y, max_y = extents
    histogram = np.zeros((bins[1], bins[0]), dtype=np.float64)  # Initialize histogram counts

    x0 = y0 = np.float64(0)

    for current_chunk_size in generate_chunk_sizes(num, chunk_size):
        points, x0, y0 = compute_trajectory_chunk(a, b, c, current_chunk_size, x0, y0)
        
        # Update histogram with current chunk
        hist, _, _ = np.histogram2d(points[:, 0], points[:, 1], bins=bins,
                                    range=[[min_x, max_x], [min_y, max_y]], density=True)
        histogram += hist

    # Normalize to get PDF
    histogram /= histogram.sum()  # This makes it a probability density function

    return histogram


def render_pdf_histogram(histogram, extents, params, color_map='hot'):
    # Render the PDF histogram as an image
    
    # Create a figure
    fig = plt.figure(figsize=(8, 8),facecolor='gainsboro')
    ax = fig.add_subplot(1, 1, 1)

    # Plotting with density color mapping
    img = ax.imshow(histogram.T, origin="lower", cmap=color_map,
              extent=[extents[0], extents[1], extents[2], extents[3]])

    # Add color bar at the bottom
    cbar = fig.colorbar(img, ax=ax, location='bottom')
    cbar.set_label('Density')  # Label for color bar
    
	#ax.axis('equal')
    plt.tight_layout()
    
    #plt.pause(1)
    #plt.close(fig)
    plt.show()


def main_pdf(image_size=(1000, 1000), color_map='hot', chunk_size=1750000):
    # Execute processes to generate and render the Hopalong Attractor PDF
    try:
        params = get_attractor_parameters()
        
        # Start the time measurement
        start_time = time.process_time()
        
        extents = compute_full_trajectory_extents(params['a'], params['b'], params['c'], params['num'])
        histogram = compute_pdf_histogram(params['a'], params['b'], params['c'], params['num'], chunk_size, extents, bins=image_size)
        render_pdf_histogram(histogram, extents, params, color_map)
        
        # End the time measurement
        end_time = time.process_time()

        # Calculate the CPU user and system time
        cpu_sys_time_used = end_time - start_time

        # Calculate the memory resources used
        memMb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024.0/1024.0
        
        print(f'CPU User&System time used: {cpu_sys_time_used:.2f} seconds')
        print (f'Memory (RAM): {memMb:.2f} MByte used')
        
    except Exception as e:
        print(f"An error occurred: {e}")


# Run PDF-based visualization
if __name__ == "__main__":
    main_pdf()

