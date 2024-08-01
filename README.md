# Hopalong Attractor in Python

The "Hopalong" attractor, invented by Barry Martin from Aston University in Birmingham, England, gained fame through A.K. Dewdney's description in the September 1986 issue of Scientific American. The German edition, Spektrum der Wissenschaft, further popularized it in Germany with a translation titled "Hüpfer" in the Computer-Kurzweil section.

## Overview
This Python program calculates and displays the "Hopalong" attractor.

It can be executed from a terminal using the following command:
```sh
python3 /path/to/my/file/hopalong.py


Requirements

To run this program, you need to have the following Python libraries installed:

numpy
matplotlib
numba
(Note: math is a standard library)
Features

This program comes in two versions:

Basic Hopalong Version: Calculates and displays the Hopalong attractor.
Advanced Version: This version additionally tracks the pixel hit count (density) to control the rendering via the colormap and generates detailed statistics regarding pixel hit counts and their distribution.
Performance optimization is achieved by using the Numba @njit (nopython=True) decorator for trajectory calculation and trajectory image generation. This results in significant speed improvements. While similar algorithms in Rust are still at least twice as fast with better memory management, this Python implementation is still quite efficient.

User Input

The program prompts the user for the following parameters:

a (float or integer): A parameter of the Hopalong equation.
b (float or integer): A parameter of the Hopalong equation.
c (float or integer): A parameter of the Hopalong equation.
num (integer): The number of iterations (e.g., 1,000,000 or 1_000_000).
Example parameters: a = -2, b = -0.33, c = 0.01, num = 200,000,000

Latest Code Changes

Using the math.copysign function [copysign(1.0, x)] ensures the behavior of floating-point numbers adheres to IEEE 754 standards (signed zero). This affects how certain input parameters a, b, and c behave, particularly in borderline cases.

For example:

a = 1, b = 2, c = 3
a = 0, b = 1, c = 1
a = 1, b = 1, c = 1
However, parameters such as:

a = 0, b = 1, c = 0
a = 1, b = 0, c = 1
a = 1, b = 1, c = 0
will result in a kind of "singularity."

Current Development

The latest version includes breaking computations into chunks to optimize memory management, thus avoiding memory swapping between RAM and SSD. This improves speed, ensuring that the computing time increases proportionally with the number of iterations, which is the expected behavior.

Notes

If you select a very high value for num (the number of iterations), performance might decrease due to memory swap usage (RAM to SSD). To mitigate this, float32 was selected as the data type for the points array. However, this depends on your system and available RAM. Running 200,000,000 iterations should still be feasible.
Users on macOS 14.x with Python 3.12.x might experience crashes while interacting with the plot window. Using the specific backend TkAgg works perfectly. This should not be necessary for other operating systems.

# import matplotlib
# matplotlib.use('TkAgg')


References

John Lansdown, Rae A. Earnshaw (Editors), Computers in Art, Design and Animation, Springer-Verlag, Chapter: Graphic Potential of Recursive Functions, Barry Martin, ISBN-13: 978-1-4612-8868-8, e-ISBN-13: 978-1-4612-4538-4
Spektrum der Wissenschaft (German version of Scientific American), Computer-Kurzweil (September 1988), Computergraphik A. K. Dewdney, ISBN 3-922508-50-2