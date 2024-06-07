# hopalong_python
Calculate and display the "Hopalong" attractor

Hopalong attractors are fractals introduced by Barry Martin of Aston University in Birmingham, England.
This program can calculate and display the "Hopalong attractor"

You can run the python script from the terminal --> python3 /path/to/my/file/hopalong.py

Requires numpy and matplotlib respectively numba for the @jit version to be installed. 

Remark: With MacOs 14.x I had issues. The plot window and python crashed during interaction with the plot window.
        Using the specific Backend TkAgg works perfect for me. Should not be needed for other operating systems.

# import matplotlib
# matplotlib.use('TkAgg')

The ".py" version is considerable slower than my Rust version. If speed is essential use the "@jit" version!!!
@jit is used to compile the iterative calculation loop for the fractal. As @jit is not compatible with matplotlib the function hopalong has been splitted into the calculation and plotting part. However, the plot function is optimized by vectorization without a loop and @jit would not contribute significantly to the execution speed.

Using @jit for the calculation loop is a revelation in terms of speed...

The programm will ask for the parameters a, b, c and num (number of iterations). Num must be entered as integer 1000000 or 1_000_000. 
The parameters a, b and c can be entered in floating point or integer format.
'a' must be a non-zero value


