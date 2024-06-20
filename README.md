# hopalong_python
Calculate and display the "Hopalong" attractor

The inventor of Hopalong is Barry Martin from Aston University in Birmingham/England. A. K. Dewdney described these fractals in September 1986 in the magazine Scientific American and made them famous. The German edition Spektrum der Wissenschaft distributed Hopalong in Germany with a translation of this article in the Computer-Kurzweil section. “Hüpfer” is the German name for Hopalong (hop-along, keep hopping).

This Python programs can calculate and display the “Hopalong” attractor

You can run the Python script from the terminal --> e.g.
python3 /path/to/my/file/hopalong_x.py

Requires the installation of numpy and matplotlib and numba for the advanced version.

Note: I had problems with MacOs 14.x and Python 3.12.x. The plot window and Python crashed while interacting with the plot window. Using the specific backend TkAgg works perfectly for me. Shouldn't be necessary for other operating systems.

"# import matplotlib"
"# matplotlib.use('TkAgg')"

If speed is essential use the advanced version. Using @njit for itarative loops is a revelation in terms of speed.
The advanced version tracks the pixel 'hit count' (density) to control the colormap and generates some statistics about pixel hit-counts and their distribution.

The program asks for the parameters a, b, c and num (number of iterations). Num must be entered as an integer e.g. 1000000 or 1_000_000. The parameters a, b and c can be entered in floating point or integer format. The value of 'a' should not be zero and the input validation returns a corresponding error. You can enter a very small value instead e.g. 1e-10 

If you select a very high value for 'num' ,the number of iterations, then the performance might decrease additionally due to memory swap use (RAM>>SSD). To compensate this float32 was selected as data type for the points-array. However, this is depending on your available System and RAM.
Anyway 200_000_000 iterations is still easy with the advanced version.

Have fun!!

Remark: The hopalong_advanced_doc.py file in the documantation folder is the same as hopalong_advanced.py but with DocString documentation/comments used as a basis to create the HTML documentation with pdoc. Open the HTML file with your preferred webbrowser.

Example plots in .png format, created with the advanced version, are available in the examples folder.

Literature:

John Lansdown Rae A. Earnshaw Editors
Computers in Art, Design and Animation
Springer-Verlag
ISBN-13: 978-1-4612-8868-8,  e-ISBN-13: 978-1-4612-4538-4

Chapter: 
Graphic Potential of Recursive Functions, pages 109 - 129 
by Barry Martin

-----------------------------------------------------------------------------------

Computer Kurzweil (German version of Scientific American),
Spektrum der Wissenschaft: 
Verständliche Forschung, ISBN 3-922508-50-2, 
Kapitel 1, Computergraphik by A. K. Dewdney
