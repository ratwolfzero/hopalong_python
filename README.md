# hopalong_python
Calculate and display the "Hopalong" attractor

The inventor of Hopalong is Barry Martin from Aston University in Birmingham/England. A. K. Dewdney described these fractals in September 1986 in the magazine Scientific American and made them famous. The German edition Spektrum der Wissenschaft distributed Hopalong in Germany in the November 1986 issue with a translation of this article in the Computer-Kurzweil section. “Hüpfer” is the German name for Hopalong (hop-along, keep hopping).

This Python program can calculate and display the “Hopalong” attractor

You can run the Python script from the terminal --> e.g.
python3 /path/to/my/file/hopalong.py

Requires the installation of numpy and matplotlib or numba for the @jit version.

Note: I had problems with MacOs 14.x and Python 3.12.x. The plot window and Python crashed while interacting with the plot window. Using the specific backend TkAgg works perfectly for me. Shouldn't be necessary for other operating systems.

"# import matplotlib"
"# matplotlib.use('TkAgg')"

The ".hopalong.py" version is significantly slower than my Rust version. If speed is important, use the "@jit" version!!!

@jit is used to compile the iterative calculation loop for the fractal. Since @jit is not compatible with matplotlib, the hopalong function was split into the calculation and plotting parts. However, the plotting function is optimized by vectorization without a loop and @jit would not significantly contribute to the execution speed.

Using @jit for the calculation loop is a revelation in terms of speed...

The program asks for the parameters a, b, c and num (number of iterations). Num must be entered as an integer 1000000 or 1_000_000. The parameters a, b and c can be entered in floating point or integer format. 'a' must be a non-zero value to avoid division by zero during normalization of pixel data. You can enter a very small value e.g. 1e-10 instead.

Since I am working with a MacMini with 8 GB RAM, I tried to optimize the memory usage by using np.empty, float32 and int16, since with a number of iterations > 100_000_000* the execution speed is significantly slowed down by swapping memory to the hard disk (swap RAM >> SSD).
Initializing the vectors as np.empty should not be a problem, since all elements of the vectors should be overwritten and no random values ​​should remain. Float64 should theoretically be faster on a 64-bit machine, but I think it is negligible and as described, float32 is a bit more economical with memory. Ditto Int16. Float32 and Int16 should be sufficient in terms of code robustness, but you can also try float64, int32 or np.zero for vector initialization if you like.

*The current image size (10k x 10k) requires a higher number of iterations. If you reduce the image size and adjust the validation of the "num" value, you can work with significantly fewer iterations to generate interesting views.

