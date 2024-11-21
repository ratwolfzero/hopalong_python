
# Calculate & Visualize the Hopalong Attractor with Python

![Example Attractor Image](./examples/Figure_32.png)
  
- [Calculate \& Visualize the Hopalong Attractor with Python](#calculate--visualize-the-hopalong-attractor-with-python)
  - [Abstract](#abstract)
    - [The chosen core algorithm and the motivation for it](#the-chosen-core-algorithm-and-the-motivation-for-it)
      - [Motivation](#motivation)
      - [Core algorithm](#core-algorithm)
  - [Requirements](#requirements)
  - [Usage](#usage)
    - [Input Parameters](#input-parameters)
    - [Output](#output)
  - [Features, Functionality, and Special Scenarios](#features-functionality-and-special-scenarios)
    - [Program Variants](#program-variants)
    - [Image Pixels and Color Mapping](#image-pixels-and-color-mapping)
    - [Pixel Hit Counts (Density) and Visualization](#pixel-hit-counts-density-and-visualization)
    - [Application of Copysign (Math Module) as Signum function](#application-of-copysign-math-module-as-signum-function)
    - [Special Constellations and Edge Cases of the Attractor](#special-constellations-and-edge-cases-of-the-attractor)
    - [Optional Features](#optional-features)
  - [Performance Optimization](#performance-optimization)
    - [Just-In-Time Compilation (JIT)](#just-in-time-compilation-jit)
    - [Two-Pass Approach](#two-pass-approach)
    - [Two-Pass Code Section](#two-pass-code-section)
    - [Alternative Solutions](#alternative-solutions)
      - [One-Pass Approach with Full Trajectory Caching\*](#one-pass-approach-with-full-trajectory-caching)
      - [One-Pass Approach with Limited Memory Usage (Chunked or No Caching)\*](#one-pass-approach-with-limited-memory-usage-chunked-or-no-caching)
      - [Potentially other, more sophisticated solutions](#potentially-other-more-sophisticated-solutions)
    - [Conclusion](#conclusion)
  - [Recent Code Changes](#recent-code-changes)
  - [Enjoy the Exploration](#enjoy-the-exploration)
  - [References](#references)
    - [References for Python Libraries and Modules](#references-for-python-libraries-and-modules)

## Abstract

The "Hopalong" attractor<top>*<top>, authored by Barry Martin of Aston University in Birmingham, England, was popularized by A.K. Dewdney in the September 1986 issue of Scientific American. In Germany, it gained further recognition through a translation titled "Hüpfer" in Spektrum der Wissenschaft.  
<sub>*Nicknamed by A.K. Dewdney.</sub>

The Python programs provided calculate and visualize the “Hopalong” attractor by iterating the following recursive functions:

$$
\large
\begin{cases}
x_{n+1} = y_n - \text{sgn}(x_n) \sqrt{\lvert b x_n - c \rvert} \\
y_{n+1} = a - x_n
\end{cases}
\large
$$

Where:

- The sequence starts from the initial point (x<sub>0</sub> , y<sub>0</sub>) = (0 , 0)
- x<sub>n</sub> and y<sub>n</sub> represent the coordinates at the n-th iteration of the attractor.
- a, b and c are parameters influencing the attracto's dynamics.
- sgn is the sign (signum) function, but math.copysign() is used, which is defined as follows:

$$
\text{copysign}(1.0,x) =
\begin{cases}
1.0  & \text{if } & \text{x } & \text{is} &\text{positive}, & \text{0.0} & or &\text{infinity} \\
-1.0 & \text{if } & \text{x } & \text{is} &\text{negative}, & \text{-0.0} & or &\text{neg. infinity}
\end{cases}
$$

### The chosen core algorithm and the motivation for it

#### Motivation

- Representation of the attractor as a density map to visualize point concentration.
- Calculation with a very high number of iterations at high processing speed and low memory requirements.

#### Core algorithm

Two-pass algorithm with separate calculation of:

- The spatial extent of the attractor trajectory (first pass).

- Direct mapping of the sequentially generated floating-point values in continuous space to a discrete image grid, while tracking the number of pixel hits to generate the density map (second pass).

Just-in-time (JIT) compilation supported by a low complexity code structure.  

For further hints regarding two-pass approach, see [Two-Pass Approach](#two-pass-approach)

[Back to Table of Contents](#calculate--visualize-the-hopalong-attractor-with-python)

## Requirements  

To run the programs, the following Python libraries or Modules must be installed / imported:

- matplotlib *
- numpy *  
- numba *
<sub>(as of October 26, 2024, Numba only supports Python versions up to 3.12.7. Versions 3.13.x are currently not supported)</sub>
- math *
  
(* mandatory)

Optional (for performance tracking):

- time
- resource
  
Import the `time` and `resource` libraries if you want to track process time and system memory usage.  
Otherwise, please comment out the relevant code snippets in the import section and the main() function.
  
    ...

    #import time
    #import resource 
    
    ...
        ...
            # Start the time measurement
            #start_time = time.process_time()

            ...

            # End the time measurement
            #end_time = time.process_time()

            # Calculate the CPU user and system time
            #cpu_sys_time_used = end_time - start_time

            # Calculate the memory resources used
            #memMb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024.0/1024.0
        
            #print(f'CPU User&System time: {cpu_sys_time_used:.2f} seconds')
            #print (f'Memory (RAM): {memMb:.2f} MByte used')

        ...

[Back to Table of Contents](#calculate--visualize-the-hopalong-attractor-with-python)

## Usage

### Input Parameters

When you run the programs, you will be prompted to enter the following parameters, which are crucial for determining the behavior of the Hopalong Attractor:

- **a (float or integer)**: The first parameter affecting the attractor's dynamics.
- **b (float or integer)**: The second parameter affecting the attractor's dynamics.
- **c (float or integer)**: The third parameter affecting the attractor's dynamics.
- **n (integer)**: The number of iterations to run (e.g. 1e6, 1_000_000 or 1000000).

**Example parameters**:

- a = -2
- b = -0.33
- c = 0.01  
- n = 2e8

Experimenting with different values of these parameters will yield diverse and intricate visual patterns.

### Output

The programs produce a visual representation of the Hopalong Attractor. The image displays the trajectory as a density map, where color intensity represents the frequency of points visited. Lighter areas indicate regions of higher density. This provides a striking visual of the attractor's complex structure. The density map also allows a 3D representation of the attractor by displaying the density on the Z-axis.

Basic Version 2D

![Example Attractor Image](./examples/Figure_ex_1.png)

Basic Version 3D

![Example Attractor Image](./examples/Figure_ex_1_1.png)

Extended Version

![Example Attractor Image](./examples/Figure_ex_2.png)
[Back to Table of Contents](#calculate--visualize-the-hopalong-attractor-with-python)

## Features, Functionality, and Special Scenarios

### Program Variants

- Basic: Calculates and displays the Hopalong attractor as a 2D density map with an integrated color bar.
  
- Basic 2D/3D: Provides the functionality of the Basic variant with the option to visualize the attractor in 3D and display the normalized density on the Z axis. The visualization mode can be selected at runtime.
  
- Extended: Includes all features of the Basic version (except the color bar), along with additional statistics and a visualization of the pixel hit count distribution.  
  
Note:
The code for the Basic variant supports both 2D and 3D visualization. Simply comment out the relevant render_trajectory_image function to switch modes. Alternatively, use the 2D/3D variant, which allows you to select the visualization mode during runtime.

[See Recent Code Changes](#recent-code-changes)

Examples of outputs can be found in the "Usage" section above.

### Image Pixels and Color Mapping

In all program variants, pixels are color-coded based on the frequency of trajectory point "hits," referred to as the pixel hit count.

### Pixel Hit Counts (Density) and Visualization

Point-to-Pixel Mapping

- Trajectory points are represented as floating-point coordinates in a two-dimensional continuous space. To visualize these points as a discrete image, they must be mapped to integer pixel coordinates. This is achieved by applying scaling factors derived from the trajectory’s extents (minimum and maximum values) and the image dimensions. These scaling factors ensure that continuous coordinates are appropriately transformed to fit within the image’s pixel grid.

Integer Conversion and Density Representation:

- Floating-point coordinates are converted to integers to determine their corresponding pixel locations. This conversion is inherently lossy: closely spaced trajectory points in continuous space may map to the same pixel, resulting in multiple "hits" for that pixel.
  
- An image array is initialized with zeros. For each mapped pixel location, the hit count at the corresponding array index is incremented. Pixels with higher hit counts represent areas of greater density, approximating the local concentration of trajectory points in the continuous space. The sum of all pixel hit counts corresponds to the number of iterations.

Visualization with Colormap:

- The Matplotlib "hot" colormap is applied to represent hit counts as colors. To enhance visualization, the colormap automatically normalizes the hit counts to fit within its gradient range. Darker colors correspond to lower hit counts, while lighter colors indicate higher hit counts, creating a visual gradient that highlights areas of intense activity within the attractor.

Remarks:

- Mapping trajectory points to pixel coordinates and counting hits provides a discrete approximation of point density in continuous space. This technique effectively highlights areas of higher concentration.
  
- To clearly demonstrate this, the following two images compare the pixel-based density estimation approach with the histogram-based density approximation method. The first image shows the results of mapping trajectory points to integer pixel coordinates and counting hits, while the second image shows the output of NumPy’s histogram function, np.histogram2d(..., density=True), which estimates the density of the original floating-point trajectory data using a binning approach. Both visualizations produce quite similar results in highlighting areas of higher concentration, which are effectively captured by both methods.

- The intensity of the colormap gradient depends on the resolution of the image (number of pixels) or the number of bins in the histogram. Lower resolution or fewer bins lead to a more intense gradient because more trajectory points are concentrated within a smaller area, amplifying the density contrast.

![Example Attractor Image](./examples/Figure_ex_1.png)
![Example Attractor Image](./examples/true_PDF_histogram.png)

[Back to Table of Contents](#calculate--visualize-the-hopalong-attractor-with-python)

### Application of Copysign (Math Module) as Signum function

The programs now utilizes the math.copysign function "copysign(x,y)"  
which returns a float with the magnitude (absolute value) of x but the sign of y.  
On platforms that support signed zeros, copysign(1.0, -0.0) returns -1.0.

$$
\text{copysign}(1.0,x) =
\begin{cases}
1.0  & \text{if } & \text{x } & \text{is} &\text{positive}, & \text{0.0} & or &\text{infinity} \\
-1.0 & \text{if } & \text{x } & \text{is} &\text{negative}, & \text{-0.0} & or &\text{neg. infinity}
\end{cases}
$$

This adjustment alters the behavior of certain parameter sets, resulting in intricate patterns instead of periodic orbits or fixed point( (0, 0) with a = 0), which is the case when using the standard signum function.

Periodic orbits are trajectories in which the system returns to the same state after a fixed number of iterations.

For example, the following parameter combinations may yield complex patterns:

- a = 1, b = 2, c = 3 or  

- a = 0, b = 1, c = 1 or  

- a = 1, b =1, c = 1

### Special Constellations and Edge Cases of the Attractor

Despite using the Copysign function, some parameter sets will still lead to periodic or near-periodic orbits instead of intricate patterns, such as:

- Set 1: a = p , b = 0, c = 0  

- Set 2: a = p, b = 0, c = p

- Set 3: a = p, b = p, c = 0
  
Here, ( p ) is a constant parameter that remains the same within each of these sets.

In these cases, you may observe high-density cycles, where a relatively small number of pixels are hit repeatedly, suggesting that the system is likely settling into periodic orbits. Additionally, many of these "high-density cycle pixels" tend to be located at the boundaries of the attractor extents.

For example, with parameter set (3) the Hopalong equations are given by:

$$
\large
\begin{cases}
x_{n+1} = y_n - \text{sgn}(x_n) \sqrt{\lvert p x_n \rvert} \\
y_{n+1} = p - x_n
\end{cases}
\large
$$

and we observe the 3-cycle: (0, 0), (0, p), and (p, p). The pixel density is: n / 3

start: (x<sub>0</sub> , y<sub>0</sub>) = (0 , 0)

--> (x<sub>1</sub> , y<sub>1</sub>) = (0 , p)

--> (x<sub>2</sub> , y<sub>2</sub>) = (p , p)

--> (x<sub>3</sub> , y<sub>3</sub>) = (0 , 0), cycle completed

Example a=5, b=5, c=0 (1_200_000 iterations):
![Example Attractor Image](./examples/Figure_ex_3.png)
![Example Attractor Image](./examples/Figure_ex_4.png)
![Example Attractor Image](./examples/Figure_ex_5.png)

If you want to experiment with this, it is recommended to reduce the total number of pixels by lowering the image resolution (e.g., 100x100) to achieve a clearer visual representation of the pixels bordering the minimum and maximum extents of the trajectory.

    def main(image_size=(100, 100), color_map='hot'):

By the way, this scenario is an ideal use case for the extended version of the program with pixel hit count statistics.

[Back to Table of Contents](#calculate--visualize-the-hopalong-attractor-with-python)

### Optional Features

Execution time and resources: Measurement starts after user input and records the CPU time for the entire process, including image rendering. It also tracks the system memory used.

Note: Since interactions with the plot window, such as zooming, panning, or mouse movements, are also measured, it is recommended to close the plot window automatically. This can be achieved using the commands plt.pause(1) followed by plt.close(fig). As long as there is no interaction with the plot window, the pause time from plt.pause() is not recorded by the time.process_time() function.

        #plt.show()
        plt.pause(1)
        plt.close(fig)

[Back to Table of Contents](#calculate--visualize-the-hopalong-attractor-with-python)

## Performance Optimization  

### Just-In-Time Compilation (JIT)

The program leverages the Numba JIT just-in-time compilation for performance optimization. This avoids the overhead of Python's interpreter, providing a significant speedup over standard Python loops. JIT compilation translates Python code into machine code at runtime, allowing for more efficient execution of loops and mathematical operations.
  
Dummy Calls

Dummy calls are preliminary invocations of JIT-compiled functions that prompt the Numba compiler to generate machine code before the function is used in the main execution flow. This ensures that the function is precompiled, avoiding compilation overhead during its first actual execution. This process is akin to "eager compilation," as it occurs ahead of time, but it does not require explicit function signatures in the header.

Parallelization and Race Conditions

The parallel loop function `prange` from the Numba library is not suitable for cross-iteration dependencies, such as those encountered when calculating trajectory points of recursive functions. While it is possible to restructure the second pass to use prange for populating the image array, this could introduce race conditions—situations where multiple threads access and modify shared data simultaneously, leading to inconsistent or unpredictable results. Therefore, this approach was not implemented.

[Back to Table of Contents](#calculate--visualize-the-hopalong-attractor-with-python)

### Two-Pass Approach

By separating the extent calculation (first pass) from trajectory point mapping (second pass), this approach allows for efficient sequential processing. Knowing the overall trajectory extents in advance enables direct and efficient mapping of points to image pixels, optimizing memory usage and maintaining consistent performance.

Advantages:

- Memory Efficiency: The two-pass approach reduces memory requirements by recalculating trajectory points, eliminating the need for large-scale caching.  
  
- JIT Compatibility: The simple, sequential structure is well-suited for Just-In-Time (JIT) compilation, enhancing execution speed.  
  
- Scalability: As the number of iterations grows, the two-pass approach’s efficiency in memory usage and processing speed becomes much more advantageous.

Disadvantage: Trajectory points must be computed in both passes, but this trade-off is minimal. As the number of iterations increases, the benefits of memory efficiency and processing speed outweigh this drawback

### Two-Pass Code Section

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

[Back to Table of Contents](#calculate--visualize-the-hopalong-attractor-with-python)

### Alternative Solutions

While the two-pass approach is the primary solution, it’s valuable to consider alternative one-pass methods, each with unique trade-offs in performance, memory usage, and complexity. Here’s an overview:

#### One-Pass Approach with Full Trajectory Caching*

Description: This method computes all trajectory points in a single pass and stores them in memory, enabling efficient calculation of trajectory extents and mapping to image pixels.  

- Advantages: Leveraging NumPy’s vectorized operations, this approach efficiently computes and maps points in a single pass, potentially increasing performance.
  
- Disadvantages:  
Full caching requires substantial memory, especially for high iteration counts. This may lead to performance issues from system memory swapping or even memory overflow.

#### One-Pass Approach with Limited Memory Usage (Chunked or No Caching)*

Description: These approaches attempt to reduce memory consumption by either processing the trajectory in chunks or not caching trajectory points at all. However, since the full trajectory extents are unknown at the outset, each variation faces the same limitation: pixel mappings require recalculating because trajectory extents change (floating points in continuous space).  

Chunked: The trajectory is divided into manageable chunks, each cached temporarily.  
No Caching: Points are computed and mapped to pixels directly without storing them.  

- Advantages: Limits memory usage.

- Disadvantages: Both approaches become impractical due to the following major limitations:
Data Loss and Inaccuracy: As previously computed floating-point values are irrecoverably mapped to integer pixel coordinates, it becomes impossible to retrieve the exact values for remapping, leading to data loss and inconsistencies.  

*This also applies analogously to any versions that only process floating point values.

#### Potentially other, more sophisticated solutions

No other one-pass method solutions have been investigated or considered to date. More sophisticated solutions would also contradict the minimum complexity design approach unless a significant performance improvement in calculations with a high number of iterations makes them worth considering.

### Conclusion

Overall, the two-pass approach strikes the best balance between speed, efficiency and simplicity and is therefore ideal for attractor calculations with a high number of iterations. Although the trajectory points need to be computed in both passes, the pitfalls of alternative solutions are avoided.

[Back to Table of Contents](#calculate--visualize-the-hopalong-attractor-with-python)

## Recent Code Changes

OPTIONAL: Using a 3D graph to display pixel density (normalized) on the Z axis (Basic 3D version).  

    """
    def render_trajectory_image(image, extents, params, color_map):
        # Render the trajectory image in 3D
        # Create a meshgrid for X and Y coordinates
        x = np.linspace(extents[0], extents[1], image.shape[1])
        y = np.linspace(extents[2], extents[3], image.shape[0])
        x, y = np.meshgrid(x, y)

        # Plot with normalized density (hit count) as Z values
        z = image / np.max(image) if np.max(image) > 0 else image

        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.contourf3D(x, y, z, levels=100, cmap=color_map)

        # Customize the plot
        ax.set_title(f'Hopalong Attractor - 3D Density (Z) Plot\nParams: a={params["a"]}, b={params["b"]}, c={params["c"]}, n={params["n"]:_}')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.view_init(elev=75, azim=-95)  # Adjust angle for better view

        plt.show()
    """
[Back to Table of Contents](#calculate--visualize-the-hopalong-attractor-with-python)  

## Enjoy the Exploration

- Explore displaying the attractor in 3D by using density as the Z-axis to visualize the structure in three dimensions. You can use or experiment with `ax.contourf3D` or `ax.contour3D`.

- Experiment with different image resolutions, color maps, or ways of populating the image array beyond using the hit count to explore new visual perspectives.  

[Also check out my simpler Rust version](https://github.com/ratwolfzero/hopalong)

© Ralf Becker  
Nuernberg: November 2024

----------------------------------------------------------------------------------------------------------------------------------------------------

## References

Computers in Art, Design and Animation (J. Landsdown and R. A. Earnshaw, eds.), New York: Springer–Verlag, 1989.

Barry Martin, "Graphic Potential of Recursive Functions," in Computers in Art, Design and Animation pp. 109–129.

ISBN-13: 978-1-4612-8868-8,  e-ISBN-13: 978-1-4612-4538-4

----------------------------------------------------------------------------------------------------------------------------------------------------

A.K. Dewdney in Spektrum der Wissenschaft "Computer Kurzweil" 1988, (German version of Scientific American)

ISBN-10: 3922508502, ISBN-13: 978-3922508502

----------------------------------------------------------------------------------------------------------------------------------------------------

### References for Python Libraries and Modules

1. [NumPy Documentation](https://numpy.org/doc/stable/): NumPy is a fundamental package for scientific computing in Python.
2. [Matplotlib Documentation](https://matplotlib.org/stable/contents.html): A library for creating static, interactive, and animated visualizations.
3. [Numba Documentation](https://numba.readthedocs.io/): Numba is a just-in-time compiler for optimizing numerical computations.
4. [Python Built-in Functions](https://docs.python.org/3/library/functions.html): Overview of built-in functions available in Python.
5. [Python Math Module](https://docs.python.org/3/library/math.html): Access mathematical functions defined by the C standard.
6. [Python Time Module](https://docs.python.org/3/library/time.html#module-time): Time access and conversions.
7. [Python Resource Module](https://docs.python.org/3/library/resource.html): Interface for getting and setting resource limits.

[Back to Table of Contents](#calculate--visualize-the-hopalong-attractor-with-python)
