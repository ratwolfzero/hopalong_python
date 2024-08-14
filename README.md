# Calculate & Display the "Hopalong" Attractor with Python

This attractor, invented by Barry Martin of Aston University in Birmingham, England, was made famous by A.K. Dewdney's description in the September 1986 issue of Scientific American as the "Hopalong attractor." The German edition, Spektrum der Wissenschaft, made it even more popular in Germany with a translation entitled "Hüpfer" in the Computer-Kurzweil section.  

## Overview  

This Python program calculates and displays the hopalong attractor.

It can be run, for example, from a terminal with the following command:

python3 /path/to/my/file/hopalong_basic.py

## Requirements  

To run this program, the following Python libraries must be installed:

- numpy  

- matplotlib  

- numba  

- (math is a standard library)  

## Features  

This program comes in two versions:

Basic version: Calculates and displays the hopalong attractor.

Extended version:
The extended version additionally tracks the pixel hit count (“density”) and generates detailed statistics on the pixel hit count and its distribution.

Rendering of the image pixels is done for both versions using a color map that depends on the pixel density (number of hits).

Performance optimization by using the Numba @njit decorator*

*Alias ​​for nopython=true

By avoiding Numpy vectorization, parallelization with Numba/Numba prange (prange is not applicable for cross-iteration dependencies), parallel iteration with Python zip, the @njit decorator obviously gives the fastest results. In addition, race conditions caused by prange are avoided.

## User Input  

The program prompts the user for the following parameters:  

- a (float or integer): A parameter of the Hopalong equation.  

- b (float or integer): A parameter of the Hopalong equation.  

- c (float or integer): A parameter of the Hopalong equation.  

- num (integer): The number of iterations (e.g., 1000000 or 1_000_000).

for example try: a = -2; b = -0.33; c = 0.01; num = 200_000_000  

## Recent code changes

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

## Note for Basic and Advanced Versions

On my macOS 14.x and Python 3.12.x system, both the plot window and Python crashed during interactions with the plot window. Using the specific TkAgg or Qt5Agg backend resolved this issue. This workaround should not be necessary for other operating systems.

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
