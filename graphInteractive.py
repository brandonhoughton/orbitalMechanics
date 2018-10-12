
import math

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from matplotlib.widgets import Slider, Button, RadioButtons
from mpl_toolkits.mplot3d import Axes3D

from dataLoader import get_data


# scale, offset, (train_X, test_X, train_F, test_F, train_Y, test_Y), benchmark = get_data(shuffle=False)

# #Determine mean velocity for each planet
# planet_values = [train_X[4223 * i:4223 * (i+1),:] for i in range(8)]
# radius = [np.average(np.sqrt(val[:,0] ** 2 + val[:,1]**2)) for val in planet_values]
# velocity = [np.average(np.sqrt(val[:,2] ** 2 + val[:,3]**2)) for val in planet_values]

# pairs = zip(radius, velocity)

# avgs = [np.average(val, axis=0) for val in planet_values]
# foo = 1

# # print(np.max(train_X,axis=0))
# # print(np.min(train_X,axis=0))


# Returns the network evaluated at each point
def phi(sess, x, y, u, v):
    viz_dic = {'X:0':np.array([x, y, u, v]).T}
    op = sess.graph.get_tensor_by_name('Phi/dense/Sigmoid:0')
    return sess.run(op, feed_dict=viz_dic)

# Returns the velocity vectors for a given mass
def getVelocity(m, x, y):
    G = 1
    r = np.sqrt(np.square(x) + np.square(y))
    mag = np.sqrt((G * m)/r)

    theta = np.arctan2(y, x) + (math.pi / 2.0)

    return mag * np.cos(theta), mag * np.sin(theta)
    
def getMeshGrid(n):
    x = np.linspace(-1, 1, n)
    y = np.linspace(-1, 1, n)
    x, y = np.meshgrid(x, y)
    return np.reshape(x, (-1)), np.reshape(y, (-1))

# Returns an array of points for phi(x,y)
def f(sess, w):
    n = 200
    # Setup surface plot
    x, y = getMeshGrid(n)

    # Setup velocity
    u, v = getVelocity(w, x, y)

    # Calculate phi
    z = phi(sess, x,y,u,v)

    return (np.reshape(x,(n,n)),np.reshape(y,(n,n)), np.reshape(z,(n,n)))
    #return (x, y, np.squeeze(z))



with tf.Session(graph=tf.Graph()) as sess:

    # Load the trained model
    new_saver = tf.train.import_meta_graph('./network/400000.meta')
    new_saver.restore(sess, './network/400000')

    # Setup plot
    fig = plt.figure(num=0)
    ax = fig.add_subplot(111, projection='3d')

    # Default slider values
    w0 = 0.01
    delta_w = 0.00001

    axcolor = 'c'
    axMass = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor=axcolor)
    sMass = Slider(axMass, 'Mass', 0.00001, 0.05, valinit=w0, valstep=delta_w)


    x, y, z = f(sess, w0)

    surface = ax.plot_surface(x, y, z, alpha=0.5)


    # Add planet trajectories
    # TODO color planets individualy
    planets = np.load('./train/p400000.npy')
    ax.scatter(planets[0], planets[1], planets[2], zdir='z', c=np.squeeze(planets[2]))

    showPlanets = True

    def update(val):
        w = sMass.val
        x,y,z = f(sess, w)
        # surface.set_data(x,y)
        # surface.set_3d_properties(z)
        ax.clear()
        surface = ax.plot_surface(x, y, z,alpha=0.5)
        if (showPlanets):
            ax.scatter(planets[0], planets[1], planets[2], zdir='z', c=np.squeeze(planets[2]))

        fig.canvas.draw_idle()
    sMass.on_changed(update)

    resetax = plt.axes([0.8, 0.025, 0.1, 0.04])
    button = Button(resetax, 'Toggle Planets', color=axcolor, hovercolor='0.975')

    def reset(event,showPlanets):
        showPlanets = not showPlanets
        if showPlanets:
            ax.scatter(planets[0], planets[1], planets[2], zdir='z', c=np.squeeze(planets[2]))
    button.on_clicked(lambda event : reset(event,showPlanets))

    plt.show()