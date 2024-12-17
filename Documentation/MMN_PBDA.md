<!-- markdownlint-disable MD033 -->

# Mathematical Notation for Pixel-Based Density Approximation

Let's define the following:

* **T**: Set of trajectory points, T = {t<sub>1</sub>, t<sub>2</sub>, ..., t<sub>n</sub>}, where n is the number of trajectory points.
* **t<sub>i</sub>**: The i-th trajectory point, represented as a 2D vector in continuous space, t<sub>i</sub> = (x<sub>i</sub>, y<sub>i</sub>), where x<sub>i</sub> and y<sub>i</sub> are floating-point coordinates.
* **x<sub>min</sub>, x<sub>max</sub>, y<sub>min</sub>, y<sub>max</sub>**: Minimum and maximum x and y values of the trajectory points, respectively.
* **W, H**: Width and height of the image (in pixels).
* **S<sub>x</sub>, S<sub>y</sub>**: Scaling factors for x and y coordinates, calculated as:

  $$
  S_x = \frac{W}{x_{\text{max}} - x_{\text{min}}}, \quad S_y = \frac{H}{y_{\text{max}} - y_{\text{min}}}
  $$
  
* **p<sub>i</sub>**: The pixel coordinates corresponding to trajectory point t<sub>i</sub>, p<sub>i</sub> = (u<sub>i</sub>, v<sub>i</sub>), where u<sub>i</sub> and v<sub>i</sub> are integer pixel indices.

* **D**: The Density Heatmap Matrix is defined as:

  $$
  D_{uv} = 0, \quad \text{for all} \, u \in [0, W-1], \, v \in [0, H-1]
  $$

## 1. Continuous to Discrete Mapping

The mapping from continuous coordinates (x<sub>i</sub>, y<sub>i</sub>) to discrete pixel coordinates (u<sub>i</sub>, v<sub>i</sub>) is given by:

  $$
  u_i = \text{round}(S_x \cdot (x_i - x_{\text{min}}))
  $$
  $$
  v_i = \text{round}(S_y \cdot (y_i - y_{\text{min}}))
  $$

where `round()` represents rounding to the nearest integer.

## 2. Density Tracking

The Density Heatmap Matrix D is updated as follows:

For each trajectory point t<sub>i</sub>:

  $$
  D_{u_i, v_i} = D_{u_i, v_i} + 1
  $$

This can be expressed more formally as:

  $$
  D_{uv} = \sum_{i=1}^n \delta \Big( u - \text{round}(S_x \cdot (x_i - x_{\text{min}})) \Big) \cdot \delta \Big( v - \text{round}(S_y \cdot (y_i - y_{\text{min}})) \Big)
  $$

where δ is the Kronecker delta function:

  $$
  \delta(a, b) =
  \begin{cases}
  1 & \text{if } a = b \\
  0 & \text{if } a \neq b
  \end{cases}
  $$

## 3. Total Hit Count

The sum of all elements in the Density Heatmap Matrix equals the number of trajectory points:

  $$
  \sum_{u=0}^{W-1} \sum_{v=0}^{H-1} D_{uv} = n
  $$

This confirms that each trajectory point contributes one "hit" to the heatmap, although multiple points may hit the same pixel.
