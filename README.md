# Calculate & Display the "Hopalong" Attractor with Python

The "Hopalong *" attractor, invented by Barry Martin at Aston University in Birmingham, England, was popularized by A.K. Dewdney in the September 1986 issue of Scientific American. In Germany, it gained further recognition through a translation titled "Hüpfer" in Spektrum der Wissenschaft.  
*Nicknamed by A.K. Dewdney.

## Overview  

This Python program calculates and displays the "Hopalong" Attractor by iterating the following equations:  

$$
\begin{align}
x_n+1 & = y_n-sgn(x_n).\sqrt{∣b⋅x_n−c∣} \\
y_n+1 & = a-x_n
\end{align}
$$

## Requirements  

To run this program, the following Python libraries* must be installed:  

- *matplotlib
- *numpy  
- *numba  
- (math is a standard library)
- (time)
- (resource)  
  
"time" and "recource" only if you want to track time and memory used,  
 if not please also comment out the related code snippets in main().
  
    import matplotlib.pyplot as plt
    import numpy as np
    from numba import njit
    from math import copysign, sqrt, fabs
    #import time
    #import resource 


    def main(image_size=(1000, 1000), color_map='hot'):
    # Main execution process
    try:
    params = get_attractor_parameters()
        
    # Start the time measurement
    #start_time = time.process_time()

    extents = compute_trajectory_extents(params['a'], params['b'], params['c'], params['num'])
    image = compute_trajectory_and_image(params['a'], params['b'], params['c'], params['num'], extents, image_size)
    render_trajectory_image(image, extents, params, color_map)

    # End the time measurement
    #end_time = time.process_time()

    """
    # Calculate the CPU user and system time
    cpu_sys_time_used = end_time - start_time
    # Calculate the memory resources used
    memMb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024.0/1024.0
    print(f'CPU User&System time used: {cpu_sys_time_used:.2f} seconds')
    print (f'Memory (RAM): {memMb:.2f} MByte used')
    """
        
    except Exception as e:
        print(f'An error occurred: {e}')

## Features  

This program is available in two versions:  

- Basic Version: Calculates and displays the Hopalong attractor.  

- Extended Version: In addition to calculating and displaying the Hopalong attractor, this version tracks the pixel hit count ("density") and generates   detailed statistics on the pixel hit count and its distribution.
For both versions, the rendered image pixels are color-mapped based on pixel density (number of hits).  

- Performance optimization by using the Numba @njit decorator.  

- Using Matplotlib allows the display of an interactive plot window.  

- Measure of execution time using time.process_time(), which captures CPU user plus system time. Timing begins after parameter entry and includes image   rendering.  

The time the plot window remains open is only recorded if an interaction occurs, such as zooming or panning.  
The measured time is displayed once the plot window is closed. For precise measurement, it's recommended to automatically close the window since pause() is not recorded by “time.process_time()”.  

Note: Using "time.perf_counter()" instead of "time.process_time()" and then subtract 1 second from "cpu_sys_time_used = end_time – start_time" yield to similar results.  

    #plt.show()
    plt.pause(1)
    plt.close(fig)

## Performance Optimization  

The program leverages the Numba @njit decorator for performance optimization by enabling nopython mode. This avoids the overhead of Python's interpreter, providing a significant speedup over standard Python loops.  

Key optimizations include:

- Two-pass aproach with straight forward loops and direct ieration

- Avoiding NumPy vectorization and parallel iteration with Python’s zip in favor of direct iteration.  

- Avoiding race conditions typically associated with parallelization techniques like prange, which is generally not applicable for cross-iteration   dependencies.

A two-pass method is preferable to array caching of trajectory points because it ensures accurate and consistent scaling across the entire dataset with minimal memory requirements. This approach is particularly advantageous for large-scale computations where memory efficiency and stability are critical. By separating the extent computation from the image mapping, the two-pass method provides reliable, scalable performance without the risk of memory overflow or performance degradation (swap RAM to SSD). For small-scale computations, any performance loss is marginal because point arrays increase system utilization and processing time.

Dummy calls are made to JIT-compiled functions. This step ensures that the function is pre-compiled by the JIT compiler before it's called by the interpreter, eliminating the initial compilation overhead while executing the code.  

    @njit #njit is an alias for nopython=True
    def compute_trajectory_extents(a, b, c, num):
    # Dynamically compute and track the minimum and maximum extents of the trajectory over 'num' iterations.
    x = np.float64(0.0)
    y = np.float64(0.0)

    min_x = np.inf  # ensure that the initial minimum is determined correctly
    max_x = -np.inf # ensure that the initial maximum is determined correctly
    min_y = np.inf
    max_y = -np.inf

    for _ in range(num):
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

    # Dummy call to ensures the function is pre-compiled by the JIT compiler before it's called by the interpreter.
    _ = compute_trajectory_extents(1.0, 1.0, 1.0, 2) 


    @njit
    def compute_trajectory_and_image(a, b, c, num, extents, image_size):
    # Compute the trajectory and populate the image with trajectory points
    image = np.zeros(image_size, dtype=np.uint64)
    
    # pre-compute image scale factors
    min_x, max_x, min_y, max_y = extents
    scale_x = (image_size[0] - 1) / (max_x - min_x)
    scale_y = (image_size[1] - 1) / (max_y - min_y)
    
    x = np.float64(0.0)
    y = np.float64(0.0)
    
    for _ in range(num):
        # map trajectory points to image pixel coordinates
        px = np.uint64((x - min_x) * scale_x)
        py = np.uint64((y - min_y) * scale_y)
        # populate the image "on the fly" with each computed point
        image[py, px] += 1  # respecting row/column convention

        # Update the trajectory "on the fly"
        xx = y - copysign(1.0, x) * sqrt(fabs(b * x - c))
        yy = a-x
        x = xx
        y = yy
        
    return image
    # Dummy call to ensures the function is pre-compiled by the JIT compiler before it's called by the interpreter.
    _ = compute_trajectory_and_image(1.0, 1.0, 1.0, 2, (-1, 0, 0, 1), (2, 2))  

You are invited to browse the development folder in the github repository to understand and try out different approaches.

## User Input  

The program prompts the user for the following parameters:  

- a (float or integer): Parameter 'a' of the Hopalong equation.  
- b (float or integer): Parameter 'b' of the Hopalong equation.  
- c (float or integer): Parameter 'c' of the Hopalong equation.  
- num (integer): The number of iterations (e.g., 1000000 or 1_000_000).

Example parameters:

- a = -2  
- b = -0.33  
- c = 0.01  
- num = 200_000_000

## Recent code changes

The program now uses the math.copysign function to respect the behavior of floating point numbers according to the IEEE 754 standard, particularly signed zero. This modification handles borderline cases more effectively, allowing different behavior with inputs like:  

For example:  

- a = 1, b = 2, c = 3 or  

- a = 0, b = 1, c = 1 or  

- a = 1, b =1, c = 1  

However, certain parameter combinations like:

- a =1 , b = 0, c = 0 or  

- a = 1, b = 0, c = 1 or  

- a = 1, b = 1, c = 0,  

may result in a kind of "singularity" where the attractor does not produce complex patterns.

## Enjoy the Exploration

----------------------------------------------------------------------------------------------------------------------------------------------------

## References

Barry Martin, "Graphic Potential of Recursive Functions," in Computers in Art, Design and Animation (J. Landsdown and R. A. Earnshaw, eds.), New York: Springer–Verlag, 1989 pp. 109–129.

ISBN-13: 978-1-4612-8868-8,  e-ISBN-13: 978-1-4612-4538-4

----------------------------------------------------------------------------------------------------------------------------------------------------

A.K. Dewdney in Spektrum der Wissenschaft "Computer Kurzweil" 1988, (German version of Scientific American)

ISBN 3-922508-50-2

----------------------------------------------------------------------------------------------------------------------------------------------------

Maple help:

<https://de.maplesoft.com/support/help/maple/view.aspx?path=MathApps%2FHopalongAttractor>
