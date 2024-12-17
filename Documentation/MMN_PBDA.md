<!-- markdownlint-disable MD033 -->

# Mathematical Notation for Pixel-Based Density Approximation

Let's define the following:

* **T**: Set of trajectory points, T = {t<sub>1</sub>, t<sub>2</sub>, ..., t<sub>n</sub>}, where n is the number of trajectory points.
* **t<sub>i</sub>**: The i-th trajectory point, represented as a 2D vector in continuous space, t<sub>i</sub> = (x<sub>i</sub>, y<sub>i</sub>), where x<sub>i</sub> and y<sub>i</sub> are floating-point coordinates.
* **x<sub>min</sub>, x<sub>max</sub>, y<sub>min</sub>, y<sub>max</sub>**: Minimum and maximum x and y values of the trajectory points, respectively.
* **W, H**: Width and height of the image (in pixels).
* **S<sub>x</sub>, S<sub>y</sub>**: Scaling factors for x and y coordinates, calculated as:
  * S<sub>x</sub> = W / (x<sub>max</sub> - x<sub>min</sub>)
  * S<sub>y</sub> = H / (y<sub>max</sub> - y<sub>min</sub>)
* **p<sub>i</sub>**: The pixel coordinates corresponding to trajectory point t<sub>i</sub>, p<sub>i</sub> = (u<sub>i</sub>, v<sub>i</sub>), where u<sub>i</sub> and v<sub>i</sub> are integer pixel indices.
* **D**: The Density Heatmap Matrix, a W x H matrix initialized with zeros. D<sub>uv</sub> represents the density at pixel (u, v).

## 1. Continuous to Discrete Mapping

The mapping from continuous coordinates (x<sub>i</sub>, y<sub>i</sub>) to discrete pixel coordinates (u<sub>i</sub>, v<sub>i</sub>) is given by:

* u<sub>i</sub> = round(S<sub>x</sub> * (x<sub>i</sub> - x<sub>min</sub>))
* v<sub>i</sub> = round(S<sub>y</sub> * (y<sub>i</sub> - y<sub>min</sub>))

where `round()` represents rounding to the nearest integer.

## 2. Density Tracking

The Density Heatmap Matrix D is updated as follows:

For each trajectory point t<sub>i</sub>:

D<sub>u<sub>i</sub>,v<sub>i</sub></sub> = D<sub>u<sub>i</sub>,v<sub>i</sub></sub> + 1

This can be expressed more formally as:

D<sub>uv</sub> = Σ<sub>i=1 to n</sub> δ(u - round(S<sub>x</sub> *(x<sub>i</sub> - x<sub>min</sub>)))* δ(v - round(S<sub>y</sub> * (y<sub>i</sub> - y<sub>min</sub>)))

where δ is the Kronecker delta function:

δ(a, b) = { 1 if a = b
          { 0 if a ≠ b

## 3. Total Hit Count

The sum of all elements in the Density Heatmap Matrix equals the number of trajectory points:

Σ<sub>u=0 to W-1</sub> Σ<sub>v=0 to H-1</sub> D<sub>uv</sub> = n

This confirms that each trajectory point contributes one "hit" to the heatmap, although multiple points may hit the same pixel.
