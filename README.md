# Calculate & Display the "Hopalong" Attractor with Python

The "Hopalong" attractor, invented by Barry Martin at Aston University, was popularized by A.K. Dewdney in the September 1986 issue of Scientific American. In Germany, it gained further recognition through a translation titled "Hüpfer" in Spektrum der Wissenschaft.

## Overview  

This Python program calculates and displays the hopalong attractor.

You can run the program from a terminal using the following command:

python3 /path/to/my/file/hopalong_basic.py

## Requirements  

To run this program, the following Python libraries must be installed:

- numpy  

- matplotlib  

- numba  

- (math is a standard library)  

## Features  

This program is available in two versions:  

Basic Version: Calculates and displays the Hopalong attractor.  

Extended Version: In addition to calculating and displaying the Hopalong attractor, this version tracks the pixel hit count ("density") and generates detailed statistics on the pixel hit count and its distribution.  

For both versions, the rendered image pixels are color-mapped based on pixel density (number of hits).
Performance optimization by using the Numba @njit decorator

## Performance Optimization  

The program leverages the Numba @njit decorator for performance optimization by enabling nopython mode (nopython=True). This avoids the overhead of Python's interpreter, providing a significant speedup over standard Python loops  

Key optimizations include:

Avoiding NumPy vectorization* and parallel iteration with Python’s zip in favor of direct iteration.  

Avoiding race conditions typically associated with parallelization techniques like prange, which is generally not applicable for cross-iteration dependencies.  

*In terms of the basic approach, a two-pass method is preferable to array caching of the trajectory points because it ensures accurate and consistent scaling across the entire dataset while using little memory. This approach is particularly beneficial in large-scale computations where memory efficiency and stability are crucial. By separating the extents computation from the image mapping, the two-pass method provides reliable and scalable performance without risking the memory overflow (swap RAM-->SSD) or performance degradation that can occur with caching methods.The loss in performance for small-scale computations is marginal.

Dummy calls are made to JIT-compiled functions. This step ensures that the function is pre-compiled by the JIT compiler before it's called by the interpreter, eliminating the initial compilation overhead while executing the code.  

## User Input  

The program prompts the user for the following parameters:  

- a (float or integer): Parameter 'a' of the Hopalong equation.  

- b (float or integer): Parameter 'b' of the Hopalong equation.  

- c (float or integer): Parameter 'c' of the Hopalong equation.  

- num (integer): The number of iterations (e.g., 1000000 or 1_000_000).

Example parameters:

a = -2  

b = -0.33  

c = 0.01  

num = 200_000_000

## Recent code changes

The program now uses the math.copysign function to respect the behavior of floating point numbers according to the IEEE 754 standard, particularly signed zero. This modification handles borderline cases more effectively, allowing different behavior with inputs like:  

For example:  

a = 1, b = 2, c = 3 or  

a = 0, b = 1, c = 1 or  

a = 1, b =1, c = 1  

However, certain parameter combinations like:

a =1 , b = 0, c = 0 or  

a = 1, b = 0, c = 1 or  

a = 1, b = 1, c = 0,  

may result in a kind of "singularity" where the attractor does not produce complex patterns.

## Enjoy the Exploration

----------------------------------------------------------------------------------------------------------------------------------------------------

## References

John Lansdown Rae A. Earnshaw Editors,
Computers in Art, Design and Animation,
Springer-Verlag

Chapter:
Graphic Potential of Recursive Functions, Barry Martin

ISBN-13: 978-1-4612-8868-8,  e-ISBN-13: 978-1-4612-4538-4

----------------------------------------------------------------------------------------------------------------------------------------------------

Spektrum der Wissenschaft: (German version of Scientific American), Computer Kurzweil (September 1988),
Computergraphik A. K. Dewdney

ISBN 3-922508-50-2

----------------------------------------------------------------------------------------------------------------------------------------------------

Maple help:

<https://de.maplesoft.com/support/help/maple/view.aspx?path=MathApps%2FHopalongAttractor>
