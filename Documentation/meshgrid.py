
import matplotlib.pyplot as plt
import numpy as np

x_edges = np.array([1, 2, 3])
y_edges = np.array([4, 5, 6])

X, Y = np.meshgrid(x_edges, y_edges)

fig = plt.figure(figsize=(8, 8),facecolor='gainsboro')
plt.plot(X, Y, 'o', c='blue', markersize=8)

plt.title('Meshgrid Demo')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()


