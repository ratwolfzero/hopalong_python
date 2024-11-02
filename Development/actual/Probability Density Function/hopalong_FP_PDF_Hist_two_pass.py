import numpy as np
import matplotlib.pyplot as plt
from numba import njit
from math import copysign, sqrt, fabs

def validate_input(prompt, input_type=float, check_positive_non_zero=False, min_value=None):
    while True:
        user_input = input(prompt)
        try:
            value = input_type(user_input)
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
            print('Invalid combination of parameters.')
        else:
            break
    num = validate_input('Enter a positive integer value for "num": ', int, check_positive_non_zero=True, min_value=1000)
    return {'a': a, 'b': b, 'c': c, 'num': num}

@njit
def compute_trajectory(a, b, c, num):
    trajectory = np.zeros((num, 2), dtype=np.float64)
    x, y = 0.0, 0.0

    for i in range(num):
        trajectory[i, 0] = x
        trajectory[i, 1] = y
        xx = y - copysign(1.0, x) * sqrt(fabs(b * x - c))
        yy = a - x
        x, y = xx, yy

    return trajectory
    
def calculate_pdf(trajectory, num_bins=100):
    # Define the bins
    x_min, x_max = np.min(trajectory[:, 0]), np.max(trajectory[:, 0])
    y_min, y_max = np.min(trajectory[:, 1]), np.max(trajectory[:, 1])
    
    x_bins = np.linspace(x_min, x_max, num_bins + 1)
    y_bins = np.linspace(y_min, y_max, num_bins + 1)

    # Create a 2D histogram
    hist, _, _ = np.histogram2d(trajectory[:, 0], trajectory[:, 1], bins=[x_bins, y_bins])
    
    # Calculate PDF by normalizing the histogram
    pdf = hist / np.sum(hist)  # Normalize to get probabilities

    return pdf, x_bins, y_bins

def plot_density(trajectory, pdf, x_bins, y_bins):
    # Plot the trajectory                          
    plt.figure(figsize=(10, 10))
    plt.scatter(trajectory[:, 0], trajectory[:, 1], color='red', s=0.1, label='Trajectory Points')
    
    # Create a meshgrid for plotting the PDF
    x_center = (x_bins[:-1] + x_bins[1:]) / 2  # Midpoints for x
    y_center = (y_bins[:-1] + y_bins[1:]) / 2  # Midpoints for y
    X, Y = np.meshgrid(x_center, y_center)

    # Plot the PDF as a contour plot
    plt.contourf(X, Y, pdf.T, levels=50, cmap='hot', alpha=0.5)
    plt.colorbar(label='Probability Density')
    plt.title('Hopalong Attractor Density Estimation')
    plt.xlabel('X (Cartesian)')
    plt.ylabel('Y (Cartesian)')
    plt.axis('equal')
    plt.legend()
    plt.show()

def main():            
    try:
        params = get_attractor_parameters()
        
        # First Pass: Compute the trajectory
        trajectory = compute_trajectory(params['a'], params['b'], params['c'], params['num'])
        
        # Second Pass: Calculate PDF
        pdf, x_bins, y_bins = calculate_pdf(trajectory, num_bins=100)

        # Plot the results
        plot_density(trajectory, pdf, x_bins, y_bins)
        
    except Exception as e:
        print(f'An error occurred: {e}')

# Main execution
if __name__ == '__main__':
    main()
