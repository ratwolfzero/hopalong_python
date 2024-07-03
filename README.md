# hopalong_python
Calculate and display the "Hopalong" attractor

The inventor is Barry Martin from Aston University in Birmingham/England. A. K. Dewdney described these fractals in September 1986 in the magazine Scientific American and made them famous. The German edition Spektrum der Wissenschaft distributed "Hopalong" in Germany with a translation of this article in the Computer-Kurzweil section. “Hüpfer” is the German name for "Hopalong" (hop-along, keep hopping).

This Python programs can calculate and display the “Hopalong” attractor

You can run the Python script from a terminal --> e.g.
python3 /path/to/my/file/hopalong_advanced.py

Requires the installation of numpy, matplotlib and numba.

There are 2 versions: Basic and advanced.
The advanced version also tracks the pixel 'hit count' (density) to control the colormap and generates some statistics about pixel hit-counts and their distribution.

The program asks for the parameters a, b, c and num (number of iterations). Num must be entered as an integer e.g. 1000000 or 1_000_000. The parameters a, b and c can be entered in floating point or integer format. 

Latest code changes see Development Folder xxx_dev.py


@njit
def custom_sign(x):

"""
<br />
for floating point according IEEE 754 (e.g. like implemented in Rust)
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

For example a=1, b=2, c=3 or a=0, b=1, c=1,

but a=0, b=1, c=0 or a=1, b=0, c=1 will still end up in some kind of "singularity"

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
