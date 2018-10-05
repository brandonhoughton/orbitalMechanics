import numpy as np
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from dataLoader import get_data


scale, offset, (train_X, test_X, train_F, test_F, train_Y, test_Y), benchmark = get_data(shuffle=False)
print(np.max(train_X,axis=0))
print(np.min(train_X,axis=0))

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

# plt.axes([0.025, 0.025, 0.95, 0.95])
# plt.quiver(X, Y, U, V, R, alpha=.5)
# plt.quiver(X, Y, U, V, edgecolor='k', facecolor='None', linewidth=1)
# print(np.array([X, Y, U, V]).shape)

# plt.xlim(-1, 1)
# plt.xticks(())
# plt.ylim(-1, 1)
# plt.yticks(())

#plt.show()


for i in range(5000,500000, 5000):

    phi = np.load('./train/'+str(i)+'.npy')
    phi = np.squeeze(phi)

    graph = np.array([X,Y,phi])

    fig = plt.figure(num=0)
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X, Y, phi, zdir='z', c=phi )
    plt.savefig(str(i) + ".png")
