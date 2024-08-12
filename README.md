# Calculate & Display the "Hopalong" Attractor with Python

The "Hopalong" attractor, invented by Barry Martin from Aston University in Birmingham, England, gained fame through A.K. Dewdney's description in the September 1986 issue of Scientific American. The German edition, Spektrum der Wissenschaft, further popularized it in Germany with a translation titled "Hüpfer" in the Computer-Kurzweil section.  

## Overview  

This Python program calculates and displays the "Hopalong" attractor.  

It can be executed for example from a terminal using the following command:  

python3 /path/to/my/file/hopalong.py  

## Requirements  

To run this program, you need to have the following Python libraries installed:

- numpy  

- matplotlib  

- numba  

- (math is a standard library)  

## Features  

This program comes in two versions:

Basic hopalong
Version: Calculates and displays the Hopalong attractor.  

Advanced Version:
The advanced version additionally tracks the pixel hit count (density) to control the rendering via the colormap andgenerates detailed statistics regarding pixel hit counts and their distribution.  

Performance optimization by using the Numba @njit (nopython=true)

Avoiding Numpy vectorization, parallelization with Numba / Numba prange, parallel iteration with Python zip
is obviously the fastest solution using the @njit decorator and avoids race conditions caused by prange

## User Input  

The program prompts the user for the following parameters:  

- a (float or integer): A parameter of the Hopalong equation.  

- b (float or integer): A parameter of the Hopalong equation.  

- c (float or integer): A parameter of the Hopalong equation.  

- num (integer): The number of iterations (e.g., 1000000 or 1_000_000).

try: a = -2; b = -0.33; c = 0.01; num = 200_000_000  

## Latest code changes

Using the math.copysign function [copysign(1.0, x)]  

With this signum function, the behavior of floating point numbers according to IEEE 754 (signed zero) is respected  

and some borderline cases regarding the input parameters a, b and c,  

which otherwise do not lead to complex patterns, show a different behavior.  

For example:  

a = 1, b = 2, c = 3 or  

a = 0, b = 1, c = 1 or  

a = 1, b =1, c = 1  

however, parameters such as  

a =0 , b = 1, c = 0 or  

a = 1, b = 0, c = 1 or  

a = 1, b = 1, c = 0,  

will end up in a kind of "singularity"  

## Have fun  

## Notes regarding basic and advanced version

On my system with MacOs 14.x and Python 3.12.x the plot window and Python crashed while interacting with the plot window. Using the specific backend TkAgg  (or Qt5Agg) solved this issue. Shouldn't be necessary for other operating systems.

"# import matplotlib"
"# matplotlib.use('TkAgg')"

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
