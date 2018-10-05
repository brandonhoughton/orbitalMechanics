import numpy as np
import math
import matplotlib.pyplot as plt
    


# Visualization mesh:
nx, ny = (100, 100)

x = np.linspace(-1, 1, nx)
y = np.linspace(-1, 1, ny)
X, Y = np.meshgrid(x, y)
X = np.reshape(X, (-1))
Y = np.reshape(Y, (-1))
T = np.arctan2(Y, X) + (math.pi / 2.0)
R = 1.5 + np.sqrt((Y) ** 2 + (X) ** 2)
U, V = R * np.cos(T), R * np.sin(T)

plt.axes([0.025, 0.025, 0.95, 0.95])
plt.quiver(X, Y, U, V, R, alpha=.5)
plt.quiver(X, Y, U, V, edgecolor='k', facecolor='None', linewidth=1)
print(np.array([X, Y, U, V]).shape)

plt.xlim(-1, 1)
plt.xticks(())
plt.ylim(-1, 1)
plt.yticks(())

plt.show()