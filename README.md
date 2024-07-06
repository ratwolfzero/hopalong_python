Calculate and Display the "Hopalong" Attractor

The "Hopalong" attractor, invented by Barry Martin from Aston University in Birmingham, England, gained fame through A.K. Dewdney's description in the September 1986 issue of Scientific American. The German edition, Spektrum der Wissenschaft, further popularized it in Germany with a translation titled "Hüpfer" in the Computer-Kurzweil section.
<br />

Overview
<br />
<br />
This Python program calculates and displays the "Hopalong" attractor. It can be executed from a terminal using the following command:
<br />
for example:
<br />
python3 /path/to/my/file/hopalong_advanced.py
<br />

Requirements
<br />
<br />
To run this program, you need to have the following Python libraries installed:

numpy
<br />
matplotlib
<br />
numba
<br />
<br />
Features
<br />
<br />
This program comes in two versions:

Basic hopalong Version: Calculates and displays the Hopalong attractor.
<br />
Advanced Version: The advanced version additionally tracks the pixel hit count (density) to control the rendering via the colormap and
<br /> 
generates detailed statistics regarding pixel hit counts and their distribution.
<br />

User Input
<br />
<br />
The program prompts the user for the following parameters:
<br />

a (float or integer): A parameter of the Hopalong equation.
<br />
b (float or integer): A parameter of the Hopalong equation.
<br />
c (float or integer): A parameter of the Hopalong equation.
<br />
num (integer): The number of iterations (e.g., 1000000 or 1_000_000).

<br />

Latest code changes:


@njit
def custom_sign(x):

"""
<br />
Signum function for floating point numbers according to IEEE 754 (e.g. like implemented in Rust)
<br />
1.0 if the number is positive, +0.0 or INFINITY
<br />
-1.0 if the number is negative, -0.0 or NEG_INFINITY
<br />
NaN if the number is NaN
<br />
"""

if np.isnan(x):
<br />
    return np.nan
<br />
elif x > 0 or x == 0.0:
<br />
    return 1.0
<br />
else:
<br />
return -1.0


With this user-defined Signum function, some borderline cases regarding the input parameters a, b and c ,
which otherwise do not lead to complex patterns, show a different behavior.

For example
<br />  
a = 1, b = 2, c = 3 or
<br /> 
a = 0, b = 1, c = 1 or
<br /> 
a = 1, b =1, c = 1

<br />

however, parameters such as
<br />
a =0 , b = 1, c = 0 or 
<br />
a = 1, b = 0, c = 1 or
<br /> 
a = 1, b = 1, c = 0,
<br />  
will end up in a kind of "singularity"
<br /> 

Have fun!
---------


Notes:
------
If you select a very high value for 'num' ,the number of iterations, then the performance might decrease additionally due to memory swap use (RAM>>SSD). To compensate this float32 was selected as data type for the points-array. However, this is depending on your available System and RAM.
Anyway 200_000_000 iterations is still easy...

I had problems with MacOs 14.x and Python 3.12.x. The plot window and Python crashed while interacting with the plot window. Using the specific backend TkAgg works perfectly for me. Shouldn't be necessary for other operating systems.

"# import matplotlib"
"# matplotlib.use('TkAgg')"


----------------------------------------------------------------------------------------------------------------------------------------------------

References:
-----------

John Lansdown Rae A. Earnshaw Editors,
Computers in Art, Design and Animation,
Springer-Verlag 

Chapter: 
Graphic Potential of Recursive Functions, Barry Martin

ISBN-13: 978-1-4612-8868-8,  e-ISBN-13: 978-1-4612-4538-4

----------------------------------------------------------------------------------------------------------------------------------------------------

Spektrum der Wissenschaft: (German version of Scientific American), Computer Kurzweil, Kapitel 1, Computergraphik A. K. Dewdney

ISBN 3-922508-50-2
