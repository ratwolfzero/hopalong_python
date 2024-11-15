import matplotlib.pyplot as plt
import numpy as np
from numba import njit
from math import copysign, sqrt, fabs


def validate_input(prompt, input_type=float, check_positive_non_zero=False, min_value=None):
    # Prompt for and return user input validated by type and positive/non-zero checks.
    while True:
        user_input = input(prompt)
        try:
            # Parse input as float first to handle scientific notation
            value = float(user_input)
            
            # Ensure the input is an integer, if expected
            if input_type == int:
                if not value.is_integer():
                    print('Invalid input. Please enter an integer.')
                    continue
                value = int(value)

            # Check if input is a positive non-zero value            
            if check_positive_non_zero and value <= 0:
                print('Invalid input. The value must be a positive non-zero number.')
                continue

            # Then, check minimum value
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
            print('Invalid combination of parameters. The following combinations are not allowed:\n'
                  '- a = 0, b = 0, c = 0\n'
                  '- a = 0, b = any value, c = 0\n'
                  'Please enter different values.')
        else:
            break
    n = validate_input('Enter a positive integer value > 1000 for "n": ', int, check_positive_non_zero=True, min_value=1000)
    return {'a': a, 'b': b, 'c': c, 'n': n}


@njit #njit is an alias for nopython=True
def compute_trajectory_extents(a, b, c, n):
    # Dynamically compute and track the minimum and maximum extents of the trajectory over 'n' iterations.
    x = np.float64(0.0)
    y = np.float64(0.0)

    min_x = np.inf  # ensure that the initial minimum is determined correctly
    max_x = -np.inf # ensure that the initial maximum is determined correctly
    min_y = np.inf
    max_y = -np.inf

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
        xx = y - copysign(1.0, x) * sqrt(fabs(b * x - c))
        yy = a-x
        x = xx
        y = yy
        
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
    
    x = np.float64(0.0)
    y = np.float64(0.0)
    
    for _ in range(n):
        # map trajectory points to image pixel coordinates
        px = np.uint64((x - min_x) * scale_x)
        py = np.uint64((y - min_y) * scale_y)
        # populate the image arrayy "on the fly" with each computed point
        image[py, px] += 1  # respecting row/column convention, update # of hits

        # Update the trajectory "on the fly"
        xx = y - copysign(1.0, x) * sqrt(fabs(b * x - c))
        yy = a-x
        x = xx
        y = yy
        
    return image

# Dummy call to ensure the function is pre-compiled by the JIT compiler before it's called by the interpreter.
_ = compute_trajectory_and_image(1.0, 1.0, 1.0, 2, (-1, 0, 0, 1), (2, 2))



def render_trajectory_3d(image, extents, params, color_map):
    fig = plt.figure(figsize=(10, 10), facecolor='gainsboro')
    ax = fig.add_subplot(111, projection='3d')

    # Create a meshgrid for X and Y coordinates                    
    x = np.linspace(extents[0], extents[1], image.shape[1])
    y = np.linspace(extents[2], extents[3], image.shape[0])						
    y, x = np.meshgrid(x, y)

    # Plot with density (hit count) as Z values
    # normalize density values
    z = image / np.max(image) if np.max(image) > 0 else image
    
    ax.contourf3D(x, y, z, levels=100, cmap=color_map)
    
    # optional variants
    #ax.plot_surface(x, y, z, cmap=color_map, edgecolor='none')
    
    #ax.plot_wireframe(x, y, z, color='blue', linewidth=0.5)
    
    #z_flattened = z.flatten()
    #ax.scatter(x.flatten(), y.flatten(), z_flattened, c=z_flattened, cmap=color_map, marker='.', s=1)
         
    # Customize the plot
    ax.set_title(f'Hopalong Attractor - 3D Density Plot\nParams: a={params["a"]}, b={params["b"]}, c={params["c"]}, n={params["n"]:_}')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Density norm')
    ax.view_init(elev=45, azim=45)  # Adjust angle for better view

    plt.show()


def main(image_size=(1000, 1000), color_map='hot'):
    # Main execution process
    try:
        params = get_attractor_parameters()
        extents = compute_trajectory_extents(params['a'], params['b'], params['c'], params['n'])
        image = compute_trajectory_and_image(params['a'], params['b'], params['c'], params['n'], extents, image_size)
        render_trajectory_3d(image, extents, params, color_map)
        
    except Exception as e:
        print(f'An error occurred: {e}')


# Main execution
if __name__ == '__main__':
    main()
