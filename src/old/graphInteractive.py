import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from matplotlib.widgets import Button

from src.dataLoader.planets import get_data_
from src.dataLoader import physicsUtils

planets = [
    'earth',
    'jupiter',
    'mars',
    'mercury',
    'neptune',
    'saturn',
    'uranus',
    'venus']


# Testing to validate getVelocity aprox is reasonable
# scale, offset, (train_X, test_X, train_F, test_F, train_Y, test_Y), benchmark = get_data(shuffle=False)
# def loss(R,x,y,dx,dy):
#     x,y,dx,dy = (x,y,dx,dy)/scale
#     x,y,dx,dy = (x,y,dx,dy)+offset
    
#     u, v = physicsUtils.getVelocity(R,x,y)
#     return math.sqrt((u - dx)**2 + (v - dx)**2)

# # #Determine mean velocity for each planet
# planet_values = [train_X[4223 * i:4223 * (i+1),:] for i in range(8)]
# planet_radius = [physicsUtils.radius[planetName] for planetName in planets]
# bar = [[loss(R,x,y,dx,dy) for (x,y,dx,dy) in pList] for (pList,R) in zip(planet_values,planet_radius)]

# for z in bar:
#     print("Average error: ", sum(z)/len(z))


# radius = [np.average(np.sqrt(val[:,0] ** 2 + val[:,1]**2)) for val in planet_values]
# velocity = [np.average(np.sqrt(val[:,2] ** 2 + val[:,3]**2)) for val in planet_values]

# pairs = zip(radius, velocity)

# avgs = [np.average(val, axis=0) for val in planet_values]
# foo = 1

# # print(np.max(train_X,axis=0))
# # print(np.min(train_X,axis=0))

    
# Returns the network evaluated at each point
def phi(sess, x, y, u, v):
    viz_dic = {'X:0':np.array([x, y, u, v]).T,'Xp:0':np.array([x, y, u, v]).T}
    op = sess.graph.get_tensor_by_name('Phi/dense/Sigmoid:0')
    return sess.run(op, feed_dict=viz_dic)

def phi2(sess, X, Xp):
    viz_dic = {'X:0':np.array(X), 'Xp:0':np.array(Xp)}
    op = sess.graph.get_tensor_by_name('Phi/dense/Sigmoid:0')
    return sess.run(op, feed_dict=viz_dic)
    
def getMeshGrid(n):
    x = np.linspace(-1, 1, n)
    y = np.linspace(-1, 1, n)
    x, y = np.meshgrid(x, y)
    return np.reshape(x, (-1)), np.reshape(y, (-1))

def getVelocity(scale, offset, x, y, semimajorAxis = None):
    x, y = (x , y) / np.expand_dims(scale[:2], 1) + np.expand_dims(offset[:2],1)
    if semimajorAxis is None:
        u, v = physicsUtils.getAvgVelocity(x, y)
    else:
        u, v = physicsUtils.getVelocity(semimajorAxis, x, y)
        print(u,v)
    return ((u,v) - np.expand_dims(offset[2:4],1)) * np.expand_dims(scale[2:4], 1)

# Returns an array of points for phi(x,y)
# Include target orbit to calulate phi for eliptical paths about the sepcified
# semimajor axis
def f(sess, scale, offset, targetOrbit=None):
    n = 200
    # Setup surface plot
    x, y = getMeshGrid(n)

    # Setup velocity
    u, v = getVelocity(scale, offset, x, y, semimajorAxis=targetOrbit)

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

    # Load planets and scale values
    scale, offset, (train_Xp, _, train_X, _, _, _, _, _), benchmark = get_data_(shuffle=False)

    # Default slider values
    w0 = physicsUtils.radius['earth']
    delta_w = 0.00001

    axcolor = 'c'
    # axMass = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor=axcolor)
    # sMass = Slider(axMass, 'Mass', 0.00001, 0.05, valinit=w0, valstep=delta_w)


    x, y, z = f(sess, scale, offset, targetOrbit=w0)

    surface = ax.plot_surface(x, y, z, alpha=0.5)


    # Add planet trajectories
    # TODO color planets individually
    planets_phi = phi2(sess, train_X, train_Xp)
    ax.scatter(train_X[:,0], train_X[:,1], planets_phi, zdir='z', c=np.squeeze(planets_phi))
    #planets = np.load('./train/p400000.npy')
    #ax.scatter(planets[0], planets[1], planets[2], zdir='z', c=np.squeeze(planets[2]))
#

    # def update(val):
    #     w = sMass.val

    #     useSlider = button.get_active()
    #     if useSlider:
    #         x,y,z = f(sess, scale, offset)
    #         ax.clear()
    #         surface = ax.plot_surface(x, y, z,alpha=0.5)
    #         # if (showPlanets):
    #         #     ax.scatter(planets[0], planets[1], planets[2], zdir='z', c=np.squeeze(planets[2]))
    #         ax.scatter(train_X[:,0], train_X[:,1], planets_phi, zdir='z', c=np.squeeze(planets_phi))


    #         fig.canvas.draw_idle()
    #     else:
    #         return

    # sMass.on_changed(update)



    class Index(object):
        ind = 0

        def next(self, event):
            self.ind += 1
            i = self.ind % len(planets)
            target = physicsUtils.radius[planets[i]]
            x,y,z = f(sess, scale, offset, targetOrbit=target)
            ax.clear()
            ax.plot_surface(x, y, z,alpha=0.5)
            ax.scatter(train_X[:,0], train_X[:,1], planets_phi, zdir='z', c=np.squeeze(planets_phi))
            fig.canvas.draw_idle()
            print(target)
            
        def prev(self, event):
            self.ind -= 1
            i = self.ind % len(planets)
            target = physicsUtils.radius[planets[i]]
            x,y,z = f(sess, scale, offset, targetOrbit=target)
            ax.clear()
            ax.plot_surface(x, y, z,alpha=0.5)
            ax.scatter(train_X[:,0], train_X[:,1], planets_phi, zdir='z', c=np.squeeze(planets_phi))
            fig.canvas.draw_idle()
        
    calback = Index()
    axprev = plt.axes([0.7, 0.025, 0.1, 0.04])
    axnext = plt.axes([0.81, 0.025, 0.1, 0.04])
    bnext = Button(axnext, 'Next Planet', color=axcolor, hovercolor='0.975')
    bnext.on_clicked(calback.next)
    bprev = Button(axprev, 'Prev Planets', color=axcolor, hovercolor='0.975')
    bprev.on_clicked(calback.prev)
    

    plt.show()