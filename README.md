# Calculate & Display the "Hopalong" Attractor with Python

The "Hopalong" attractor, invented by Barry Martin from Aston University in Birmingham, England, gained fame through A.K. Dewdney's description in the September 1986 issue of Scientific American. The German edition, Spektrum der Wissenschaft, further popularized it in Germany with a translation titled "Hüpfer" in the Computer-Kurzweil section.  

## Overview  

This Python program calculates and displays the "Hopalong" attractor.  

It can be executed for example from a terminal using the following command:  

python3 /path/to/my/file/hopalong.py  

## Requirements  

To run this program, you need to have the following Python libraries installed:

numpy  

matplotlib  

numba  

(math is a standard library)  

## Features  

This program comes in two versions:

Basic hopalong Version: Calculates and displays the Hopalong attractor.  

Advanced Version: The advanced version additionally tracks the pixel hit count (density) to control the rendering via the colormap and
generates detailed statistics regarding pixel hit counts and their distribution.  

Performance optimization by using the Numba @njit (nopython=true) decorator for trajectory calculation and trajectory image generation,
this is really a revelation in terms of speed!  

Using similar algorithms my Rust version is still at least twice as fast and the memory management is a dream...but still not bad for Python...  

## User Input  

The program prompts the user for the following parameters:  

a (float or integer): A parameter of the Hopalong equation.  

b (float or integer): A parameter of the Hopalong equation.  

c (float or integer): A parameter of the Hopalong equation.  

num (integer): The number of iterations (e.g., 1000000 or 1_000_000).

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

## Current development  

Version with breaking computations into chunks to optimize memory management,
thus avoiding memory swapping RAM--SSD.  

Increase in speed.
The computing time with high number of iterations (num) now increases proportionally to the number of iterations, which corresponds to the expected behavior  

## Have fun  

## Notes  
  
If you select a very high value for 'num' ,the number of iterations, then the performance might decrease additionally due to memory swap use (RAM>>SSD). To compensate this float32 was selected as data type for the points-array. However, this is depending on your available System and RAM.
Anyway 200_000_000 iterations is still easy...

I had problems with MacOs 14.x and Python 3.12.x. The plot window and Python crashed while interacting with the plot window. Using the specific backend TkAgg works perfectly for me. Shouldn't be necessary for other operating systems.

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
