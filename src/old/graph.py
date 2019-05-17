import numpy as np
import math
import matplotlib.pyplot as plt

from src.dataLoader.planets import get_data


scale, offset, (train_X, test_X, train_F, test_F, train_Y, test_Y), benchmark = get_data(shuffle=False)

#Determine mean velocity for each planet
planet_values = [train_X[4223 * i:4223 * (i+1),:] for i in range(8)]
radius = [np.average(np.sqrt(val[:,0] ** 2 + val[:,1]**2)) for val in planet_values]
velocity = [np.average(np.sqrt(val[:,2] ** 2 + val[:,3]**2)) for val in planet_values]

mv^2/r = f

pairs = zip(radius, velocity)

avgs = [np.average(val, axis=0) for val in planet_values]
foo = 1

# print(np.max(train_X,axis=0))
# print(np.min(train_X,axis=0))

# Visualization mesh:
nx, ny = (20, 20)

x = np.linspace(-1, 1, nx)
y = np.linspace(-1, 1, ny)
X, Y = np.meshgrid(x, y)
X = np.reshape(X, (-1))
Y = np.reshape(Y, (-1))
R = np.sqrt((Y) ** 2 + (X) ** 2)
T = np.arctan2(Y, X) + (math.pi / 2.0)

U, V = R * np.cos(T), R * np.sin(T)

# plt.axes([0.025, 0.025, 0.95, 0.95])
# plt.quiver(X, Y, U, V, R, alpha=.5)
# plt.quiver(X, Y, U, V, edgecolor='k', facecolor='None', linewidth=1)
# print(np.array([X, Y, U, V]).shape)

# plt.xlim(-1, 1)
# plt.xticks(())
# plt.ylim(-1, 1)
# plt.yticks(())

# plt.show()



for i in range(5000, 500000, 5000):

    phi = np.load('./train/'+str(i)+'.npy')
    phi = np.squeeze(phi)

    planets = np.load('./train/p' + str(i) + '.npy')

    graph = np.array([X,Y,phi])

    fig = plt.figure(num=0)
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X, Y, phi, zdir='z', c=phi)
    ax.scatter(planets[0], planets[1], planets[2], zdir='z', c='red')
    plt.savefig(str(i) + ".png")
