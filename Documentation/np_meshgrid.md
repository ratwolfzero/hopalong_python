# np.meshgrid

x_edges = np.array([1, 2, 3])

y_edges = np.array([4, 5, 6])

Then:

X, Y = np.meshgrid(x_edges, y_edges)

Results in:

X = [[1, 2, 3],  # x-coordinates repeated for each row
     [1, 2, 3],  
     [1, 2, 3]]

Y = [[4, 4, 4],  # y-coordinates repeated for each column
     [5, 5, 5],  
     [6, 6, 6]]

Visualization

The output grids X and Y define a grid of points:

(1,4),(2,4),(3,4)
(1,5),(2,5),(3,5)
(1,6),(2,6),(3,6)
