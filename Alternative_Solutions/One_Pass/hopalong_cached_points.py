import matplotlib.pyplot as plt
import numpy as np
from numba import njit
from math import copysign, sqrt, fabs
import time
import resource

def validate_input(prompt, input_type=float, check_positive_non_zero=False, min_value=None):
    # Prompt for and return user input validated by type and positive/non-zero checks
    while True:
        user_input = input(prompt)
        try:
            value = input_type(user_input)
            if check_positive_non_zero and value <= 0:
                print("Invalid input. The value must be a positive non-zero number.")
                continue
            if min_value is not None and value < min_value:
                print(f"Invalid input. The value should be at least {min_value}.")
                continue
            return value
        except ValueError:
            print(f"Invalid input. Please enter a valid {input_type.__name__} value.")


def get_attractor_parameters():
    a = validate_input('Enter a float value for "a": ', float)
    b = validate_input('Enter a float value for "b": ', float)
    while True:
        c = validate_input('Enter a float value for "c": ', float)
        if (a == 0 and b == 0 and c == 0) or (a == 0 and c == 0):
            print("Invalid combination of parameters. The following combinations are not allowed:\n"
                  "- a = 0, b = 0, c = 0\n"
                  "- a = 0, b = any value, c = 0\n"
                  "Please enter different values.")
        else:
            break
    num = validate_input('Enter a positive integer value for "num": ', int, check_positive_non_zero=True, min_value=1000)
    return {'a': a, 'b': b, 'c': c, 'num': num}


@njit
def compute_trajectory_and_image_cached(a, b, c, num, image_size):
    # Initialize extents
    min_x = min_y = np.inf
    max_x = max_y = -np.inf
    
    # Initialize cache for trajectory points
    trajectory_points = np.zeros((num, 2), dtype=np.float64)
    
    x = y = np.float64(0)
    
    for i in range(num):
        # Track the min and max extents
        min_x = min(min_x, x)
        max_x = max(max_x, x)
        min_y = min(min_y, y)
        max_y = max(max_y, y)
        
        # Cache the current trajectory point
        trajectory_points[i, 0] = x
        trajectory_points[i, 1] = y
        
        # Update the trajectory "on the fly"
        xx, yy = y - copysign(1.0, x) * sqrt(fabs(b * x - c)), a - x
        x, y = xx, yy
    
    # Pre-compute image scale factors using final extents
    scale_x = (image_size[1] - 1) / (max_x - min_x)
    scale_y = (image_size[0] - 1) / (max_y - min_y)
    
    # Initialize image
    image = np.zeros(image_size, dtype=np.uint64)
    
    # Use cached points to populate the image
    for i in range(num):
        x, y = trajectory_points[i, 0], trajectory_points[i, 1]
        px = np.uint64((x - min_x) * scale_x)
        py = np.uint64((y - min_y) * scale_y)
        image[py, px] += 1  # respecting row/column convention
    
    return image, (min_x, max_x, min_y, max_y)

# Dummy call to ensure the function is pre-compiled by the JIT compiler before it's called by the interpreter.
_ = compute_trajectory_and_image_cached(1.0, 1.0, 1.0, 2, (1, 1))


def render_trajectory_image(image, extents, params, color_map):
    # Render the trajectory image                                        
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(1, 1, 1, aspect='auto')
    # origin="lower" align according cartesian coordinates
    ax.imshow(image, origin="lower", cmap=color_map, extent=extents, interpolation=None)
    ax.set_title("Hopalong Attractor@ratwolf@2024\nParams: a={a}, b={b}, c={c}, num={num:_}".format(**params))
    ax.set_xlabel('X (Cartesian)')
    ax.set_ylabel('Y (Cartesian)')

    plt.show()


def main(image_size=(1000, 1000), color_map='hot'):
    # Main execution process
    try:
        params = get_attractor_parameters()

        # Start the CPU&system time measurement
        start_time = time.process_time()

        image, extents = compute_trajectory_and_image_cached(params['a'], params['b'], params['c'], params['num'], image_size)
        render_trajectory_image(image, extents, params, color_map)

        # End the CPU&system time measurement
        end_time = time.process_time()
        # Calculate the CPU&system time and memory used
        cpu_sys_time_used = end_time - start_time
        memMb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024.0/1024.0
        print(f"CPU & System time used: {cpu_sys_time_used:.2f} seconds")
        print (f"Memory (RAM): {memMb:.2f} MByte used")

    except Exception as e:
        print(f"An error occurred: {e}")


# Main execution––
if __name__ == "__main__":
    main()
