# Calculate & Display the "Hopalong" Attractor with Python

## Abstract  

The "Hopalong *" attractor, invented by Barry Martin at Aston University in Birmingham, England, was popularized by A.K. Dewdney in the September 1986 issue of Scientific American. In Germany, it gained further recognition through a translation titled "Hüpfer" in Spektrum der Wissenschaft.  
<sub>*Nicknamed by A.K. Dewdney.</sub>

This Python program calculates and displays the "Hopalong" Attractor by iterating the following system of interrelated equations (1) and (2):

$$
\begin{align}
x_n+1 & = y_n-sgn(x_n)\times\sqrt{∣b\times x_n−c∣} &(1) \\
y_n+1 & = a-x_n &(2)
\end{align}
$$

Where:

- x<sub>n</sub> and y<sub>n</sub> represent the coordinates at the nth iteration.
- a, b, c are user defined parameters that shape the attractor
- The sequence starts from an initial point (x<sub>0</sub>, y<sub>0</sub>)

...........................................................................................................................................................

### Chosen Solution and Motivation

A two-pass algorithm is employed to compute the Hopalong Attractor by sequential processing in both passes through straightforward loops.

- In the first pass, the algorithm determines the overall trajectory extents, which consist of the minimum and maximum values of the attractor trajectory.

- In the second pass, the algorithm generates the sequence of trajectory points and maps them directly to image pixel coordinates, representing the attractor hit pattern (pixel value > 0). This hit information is updated and stored in an image array, which was initialized with zero values.

The program uses Matplotlib to render the attractor, allowing interactive exploration. Supporting a very high number of iterations with low memory footprint and optimal, consistent processing speed to enable higher image resolutions. Implementing a simple program with minimal complexity, allowing optimal use of Just-In-Time (JIT) compilation
  

## Requirements  

To run this program, the following Python libraries or Modules must be installed / imported (* mandatory):

- matplotlib *
- numpy *  
- numba *  
- math *

Optional (for performance tracking):

- time
- resource
  
Import the "Time" and "Resource" libraries if you want to track process time and system memory used.
Otherwise, please comment out the relevant code snippets.
  
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
    # start_time = time.process_time()

    extents = compute_trajectory_extents(params['a'], params['b'], params['c'], params['num'])
    image = compute_trajectory_and_image(params['a'], params['b'], params['c'], params['num'], extents, image_size)
    render_trajectory_image(image, extents, params, color_map)

    # End the time measurement
    # end_time = time.process_time()

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

## Usage

### Input Parameters

Upon running the program, you'll be prompted to enter the following parameters:

- a (float or integer): Parameter 'a' of the Hopalong equation.  
- b (float or integer): Parameter 'b' of the Hopalong equation.  
- c (float or integer): Parameter 'c' of the Hopalong equation.  
- num (integer): The number of iterations (e.g., 1000000 or 1_000_000).

Example parameters:

- a = -2  
- b = -0.33  
- c = 0.01  
- num = 200_000_000

These parameters directly influence the appearance of the attractor, with different values yielding different patterns.

### Output

The program generates a visual representation of the Hopalong Attractor. The resulting image displays the trajectory where colors represent the density of hits (i.e., how often a particular point was visited).

#### Basic Version

![Example Attractor Image](./examples/Figure_ex_1.png)

#### Extended Version

![Example Attractor Image](./examples/Figure_ex_2.png)

## Features

This program is available in two versions:

- Basic version: Calculation and display of the Hopalong attractor.
- Advanced version: Like the basic version plus statistics and visualization of the pixel hit counts distribution.

Example of outputs,  see section Usage above.

### Image Pixel and Color Mapping  

In both versions of the program (basic or extended), pixels are color-coded based on the number of times they are "hit" by trajectory points, referred to as the "pixel hit count." However, trajectory points are floating-point values and do not directly correspond to pixel coordinates. Instead, they are mapped to integer pixel coordinates on the image. The mapping is handled by scale factors using the image size and trajectory extents (min, max values). For the very details you can consult the function "compute_trajectory_and_image" in the code.

### Pixel Hit Counts and Handling High Density

As each trajectory point is generated, it is mapped to corresponding pixel coordinates by converting its floating-point values into integers. This mapping process often results in certain pixels being "hit" multiple times, creating areas of varying density within the image. Initially, the image array is set to zero, and each time a pixel is hit, its value is incremented, reflecting the number of trajectory points that correspond to that pixel.

Pixels with higher hit counts are color-coded to represent their density, with the program utilizing Matplotlib's 'hot' colormap. This colormap creates a gradient that transitions from dark (indicating low hit counts) to bright (indicating high hit counts), effectively visualizing areas of higher activity within the attractor.

To ensure effective visualization, Matplotlib applies normalization to scale hit counts within the finite range of colors provided by the colormap. Pixels exceeding the defined maximum are mapped to the brightest color, guaranteeing that regions of extreme density are distinctly represented. This clear color gradient allows users to easily discern patterns of activity and better understand the Hopalong attractor's behavior.

### Application of Copysign (Math Module) as Signum function

Signum Function:  
The program now utilizes the math.copysign function "copysign(x,y)"  
Return a float with the magnitude (absolute value) of x but the sign of y.  
On platforms that support signed zeros, copysign(1.0, -0.0) returns -1.0.

$$
copysign(1.0,x) =\begin{cases}
1.0  & if & x & is &positive, & +0.0 & or &INFINITY \\
-1.0 & if & x & is &negative, & -0.0 & or &NEG. INFINITY
\end{cases}
$$

This adjustment improves handling of edge cases, allowing for different behavior. For example:

- a = 1, b = 2, c = 3 or  

- a = 0, b = 1, c = 1 or  

- a = 1, b =1, c = 1  

However, certain parameter combinations such as:

- a =1 , b = 0, c = 0 or  

- a = 1, b = 0, c = 1 or  

- a = 1, b = 1, c = 0,  

may lead to "singularities" where the attractor doesn't produce complex patterns.

### Optional Features  

Execution time* and resources: Starts after user input and measures the CPU time for the entire process including image rendering and shows the system memory used.

*Since interactions with the plot window, e.g. zooming, panning, mouse movements, are measured, it is recommended to close the plot window automatically.
This can be done, for example, by using the commands plt.pause(1) followed by plt.close(fig).
As long as there is no interaction with the plot window, the "plt.pause() time" is not recorded by the "time.process_time()" function used.

    #plt.show()
    plt.pause(1)
    plt.close(fig)

## Performance Optimization  

### Just-In-Time Compilation (JIT)

The program leverages the Numba JIT just-in-time compilation for performance optimization. This avoids the overhead of Python's interpreter, providing a significant speedup over standard Python loops.  
  
### Dummy Calls

For JIT-compiled functions dummy calls are made. This step ensures that the function is precompiled before it is called by the interpreter, thus avoiding compilation overhead the first time the code is executed.  

### Race Conditions  

 Prange, is generally not applicable for cross-iteration dependencies as it is the case when calculating the trajectory points. A separate function to populate the image array in a parallel loop using prange is possible but leads to race conditions resulting in inconsistent pixel hit rate and was therefore not implemented.

### Two-Pass Approach

By separating the extent calculation (first pass) from trajectory point mapping (second pass), this approach allows for efficient sequential processing. Knowing the overall trajectory extents in advance enables direct and efficient mapping of points to image pixels, optimizing memory usage and maintaining consistent performance.

- Memory Efficiency: The two-pass approach reduces memory requirements by recalculating trajectory points, eliminating the need for large-scale caching.  
  
- JIT Compatibility: The simple, sequential structure is well-suited for Just-In-Time (JIT) compilation, enhancing execution speed.  
  
- Scalability: As the number of iterations grows, the two-pass approach’s efficiency in memory usage and processing speed becomes much more advantageous.

Disadvantage:  
Trajectory points must be recomputed in both passes, but the impact of this trade-off is quite small and only occurs for a smaller number of iterations. As mentioned above, as the number of iterations increases, the efficiency of the two-pass approach becomes much more advantageous in terms of memory usage and processing speed.

### Alternative Solutions

#### One-Pass Approach with Caching

- Description: Trajectory points are calculated only once, stored in an array and available for further processing such as mapping points to pixels.
- Disadvantages: Requires large memory resources depending on the number of iterations and can lead to performance degradation due to system memory swapping.

#### Chunked One-Pass Approach

- Description: Trajectory points are processed in smaller segments (chunks) while caching points to manage memory usage.
- Disadvantages: While it keeps memory consumption low, this method adds complexity and overhead, often resulting in performance that is similar to or slower than the two-pass method.

#### One-Pass Approach without Caching

- Description: This method attempts to calculate and map points in a single loop (pass) without storing previous points.
- Disadvantages: Requires continuously recalculating the mapping of trajectory points to image pixels every time the trajectory extent changes making it complicated and ineffective and difficult to ensure accurate pixel mapping

Possible other, more sophisticated solutions were not taken into consideration

### Conclusion

Overall, the two-pass approach strikes the best balance of speed, efficiency, and simplicity, making it ideal for high-iteration calculations of the Hopalong Attractor. Despite the need to recalculate trajectory points, it avoids the pitfalls of alternative solutions.

### Two-Pass Code Section

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

    # Dummy call to ensure the function is pre-compiled by the JIT compiler before it's called by the interpreter.
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
        # populate the image array "on the fly" with each computed point
        image[py, px] += 1  # respecting row/column convention, update # of hits

        # Update the trajectory "on the fly"
        xx = y - copysign(1.0, x) * sqrt(fabs(b * x - c))
        yy = a-x
        x = xx
        y = yy
        
    return image
    # Dummy call to ensure the function is pre-compiled by the JIT compiler before it's called by the interpreter.
    _ = compute_trajectory_and_image(1.0, 1.0, 1.0, 2, (-1, 0, 0, 1), (2, 2)) 

## Recent Code Changes

Preparation to utilize a Color Bar to indicate the Pixel Density

    img=ax.imshow(image, origin='lower', cmap=color_map, extent=extents, interpolation='none')  # modification 'img=ax.imshow' to apply 'colorbar'
    #...
    #cbar = fig.colorbar(img, ax=ax, extend='both') # prepared to apply 'colorbar'
    #cbar.set_label('Pixel Density')

## Enjoy the Exploration

Experiment with different color maps or populate the image array differently than based on the hit count and explore new visual views.

----------------------------------------------------------------------------------------------------------------------------------------------------

## References

Barry Martin, "Graphic Potential of Recursive Functions," in Computers in Art, Design and Animation (J. Landsdown and R. A. Earnshaw, eds.), New York: Springer–Verlag, 1989 pp. 109–129.

ISBN-13: 978-1-4612-8868-8,  e-ISBN-13: 978-1-4612-4538-4

----------------------------------------------------------------------------------------------------------------------------------------------------

A.K. Dewdney in Spektrum der Wissenschaft "Computer Kurzweil" 1988, (German version of Scientific American)

ISBN-10: 3922508502, ISBN-13: 978-3922508502

----------------------------------------------------------------------------------------------------------------------------------------------------

Maple help:

<https://de.maplesoft.com/support/help/maple/view.aspx?path=MathApps%2FHopalongAttractor>

----------------------------------------------------------------------------------------------------------------------------------------------------

### References for Python Libraries and Modules

1. [NumPy Documentation](https://numpy.org/doc/stable/): NumPy is a fundamental package for scientific computing in Python.
2. [Matplotlib Documentation](https://matplotlib.org/stable/contents.html): A library for creating static, interactive, and animated visualizations.
3. [Numba Documentation](https://numba.readthedocs.io/): Numba is a just-in-time compiler for optimizing numerical computations.
4. [Python Built-in Functions](https://docs.python.org/3/library/functions.html): Overview of built-in functions available in Python.
5. [Python Math Module](https://docs.python.org/3/library/math.html): Access mathematical functions defined by the C standard.
6. [Python Time Module](https://docs.python.org/3/library/time.html#module-time): Time access and conversions.
7. [Python Resource Module](https://docs.python.org/3/library/resource.html): Interface for getting and setting resource limits.
