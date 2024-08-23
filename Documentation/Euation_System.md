
This Python program calculates and displays the "Hopalong" Attractor by iterating the following system of interrelated equations (1) and (2):

$$
\begin{align}
x_n+1 & = y_n-sgn(x_n)\times\sqrt{∣b\times x_n−c∣} & (1) \\
y_n+1 & = a-x_n &(2)
\end{align}
$$

The sequence of (x<sub>1</sub>, y<sub>1</sub>), (x<sub>2</sub>, y<sub>2</sub>), ..., (x<sub>n</sub>, y<sub>n</sub>)  coordinates is specified by an initial point (x<sub>0</sub>, y<sub>0</sub>) and three constants a, b, and c. 