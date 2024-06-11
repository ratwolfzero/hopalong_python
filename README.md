# hopalong_python
Calculate and display the "Hopalong" attractor

The inventor of Hopalong is Barry Martin from Aston University in Birmingham/England. A. K. Dewdney described these fractals in September 1986 in the magazine Scientific American and made them famous. The German edition Spektrum der Wissenschaft distributed Hopalong in Germany in the November 1986 issue with a translation of this article in the Computer-Kurzweil section. “Hüpfer” is the German name for Hopalong (hop-along, keep hopping).

This Python programs can calculate and display the “Hopalong” attractor

You can run the Python script from the terminal --> e.g.
python3 /path/to/my/file/hopalong_x.py

Requires the installation of numpy and matplotlib and numba for the advanced version.

Note: I had problems with MacOs 14.x and Python 3.12.x. The plot window and Python crashed while interacting with the plot window. Using the specific backend TkAgg works perfectly for me. Shouldn't be necessary for other operating systems.

"# import matplotlib"
"# matplotlib.use('TkAgg')"

There are 2 versions of the programm basic and advanced. The basic version is quite slow.

In the advanced version @jit is used to compile the calculation for the hopalong attractor and the routine to set the pixels and their density (hit count) which is used for the color sheme in this case.

Using @jit is a revelation in terms of speed...

The program asks for the parameters a, b, c and num (number of iterations). Num must be entered as an integer e.g. 1000000 or 1_000_000. The parameters a, b and c can be entered in floating point or integer format. The value of 'a' should not be zero and the input validation returns a corresponding error. You can enter a very small value instead e.g. 1e-10 

If you select a very high value for 'num' ,the number of iterations, then the performance might decrease additionally due to memory swap use (RAM>>SSD). To compensate this float32 was selected as data type for the pixel-array. Anyway 100_000_000 iterations are still easy with the advanced version.

Of course you can experiment with different colormaps (cmap) but for the advanced version where the pixel density is considered you should use one of the sequential colormaps.

Have fun.

Idea for statistics (advanced version): Histogram for pixel density (hits)

    ...

    plt.title(
        f"Hopalong Attractor@ratwolf2024\nParams: a={a}, b={b}, c={c}, num={num:_}")
    ------------------------------------------------------
    uniques, counts = np.unique(img, return_counts=True)
    plt.figure(figsize=(10, 10))
    plt.bar(uniques, counts, log=True)
    ------------------------------------------------------
    plt.show()

    ...


