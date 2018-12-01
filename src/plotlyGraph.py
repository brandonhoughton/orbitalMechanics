#!/usr/bin/python3
import plotly                   # RAH
import plotly.graph_objs as go  # RAH
import itertools

import numpy as np
import tensorflow as tf
#import matplotlib.pyplot as plt
#from matplotlib.widgets import Slider, Button, RadioButtons
#from mpl_toolkits.mplot3d import Axes3D

from dataLoader.planets import get_data, get_data_
from dataLoader import physicsUtils

from scipy.optimize import curve_fit
from scipy.constants import G


planets = [
    'earth',
    'jupiter',
    'mars',
    'mercury',
    'neptune',
    'saturn',
    'uranus',
    'venus',
    'none']

colorscale = [
'rgb(167, 119, 12)',
'rgb(197, 96, 51)',
'rgb(217, 67, 96)',
'rgb(221, 38, 163)',
'rgb(196, 59, 224)',
'rgb(153, 97, 244)',
'rgb(95, 127, 228)',
'rgb(40, 144, 183)',
'rgb(15, 151, 136)',
'rgb(39, 153, 79)',
'rgb(119, 141, 17)',
'rgb(167, 119, 12)']

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
    viz_dic = {'X:0':np.array([x, y, u, v]).T}
    #viz_dic = {'X:0':np.array([x, y, u, v]).T,'Xp:0':np.array([x, y, u, v]).T}
    op = sess.graph.get_tensor_by_name('Phi/dense/Sigmoid:0')
    return sess.run(op, feed_dict=viz_dic)

def phi2(sess, X):
    viz_dic = {'X:0':np.array(X)}
    #viz_dic = {'X:0':np.array(X),'Xp:0':np.array(Xp)}
    op = sess.graph.get_tensor_by_name('Phi/dense/Sigmoid:0')
    #op = sess.graph.get_tensor_by_name('pred_loss/dense/Sigmoid:0')
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

    #z = np.ones_like(x)

    # Setup velocity
    if targetOrbit is not None:
        radius = physicsUtils.radius[targetOrbit]
    else:
        radius = None

    u, v = getVelocity(scale, offset, x, y, semimajorAxis=radius)

    # Calculate phi
    z = phi(sess, x,y,u,v)

    return (np.reshape(x,(n,n)),np.reshape(y,(n,n)), np.reshape(z,(n,n)))
    #return (x, y, np.squeeze(z))

def f2(planetID, planetX, planetY, planetvX, planetvY, planetPhi, targetOrbit = None):
    planetID = np.reshape(planetID, (None))
    planetX = np.reshape(planetX, (None))
    planetY = np.reshape(planetY, (None))
    planetvX = np.reshape(planetvX, (None))
    planetvY = np.reshape(planetvY, (None))

    planetPhi = np.reshape(planetPhi, (None))

    m1 = np.array([physicsUtils.mass[id] for id in planet_ID])
    m2 = physicsUtils.mass_sun

    # Fit 1/r function
    planetR = np.sqrt(planetX ** 2 + planetY ** 2)
    def hyper(r, A, B):
        return 1/(A*r) + B

    # Hamiltonian
    def h(m1, x, y, vx, vy):
        return (m2 / 2) * (vx ** 2 + vy ** 2) - G * m1 * m2 / np.sqrt(x ** 2, + y ** 2)

    # Angular Momentum
    def l(x, y, vx, vy):
        return m2 * (x * vy - y * vx)

    hamiltonian = h(m1, planetX, planetY, planetvX, planetvY)

    momentum = l(planetX, planetY, planetvX, planetvY)

    indep = np.stack([hamiltonian, momentum], axis=-1).T

    def liniar(x, a, b, c):
        return a + b*x[0] + c*x[1]

    popt, pconv = curve_fit(liniar, indep, planetPhi)

    print("Covariance: ", pconv)
    print("Avg: ", sum(pconv)/len(pconv))

    n = 200

    if targetOrbit is not None:
        radius = physicsUtils.radius[targetOrbit]
    else:
        radius = None

    # Setup surface plot
    x, y = getMeshGrid(n)
    u, v = getVelocity(scale, offset, x, y, semimajorAxis=radius)

    if (targetOrbit is None):
        avg_m1 = sum(physicsUtils.mass.values())/len(physicsUtils.mass)
    else:
        avg_m1 = physicsUtils.mass[targetOrbit]
    hamiltonian = h(avg_m1, x, y, u, v)

    momentum = l(x, y, u, v)

    indep_grid = np.stack([hamiltonian, momentum], axis=-1).T

    #r = np.sqrt(x ** 2 + y ** 2)

    # Calculate z
    z = liniar(indep_grid, popt[0], popt[1], popt[2]).T

    # Don't messup autoscaling
    # z[z > 0.8] = np.nan

    return (np.reshape(x,(n,n)),np.reshape(y,(n,n)), np.reshape(z,(n,n)))
    




with tf.Session(graph=tf.Graph()) as sess:

    # Load the trained model
    new_saver = tf.train.import_meta_graph('network/400000.meta')
    new_saver.restore(sess, 'network/400000')


    # Load planets and scale values
    scale, offset, (train_X, _, _, _, _, _), benchmark = get_data(shuffle=False)
    #scale, offset, (train_Xp, _, train_X, _, _, _, _, _), benchmark = get_data_(shuffle=False)

    # For each planet, plot the values to an interactive <planet>.html
    for planet in planets:
        print ("Planet:",planet)
        if planet != 'none':
            w0 = physicsUtils.radius[planet]
            #continue
        else:
            w0 = None
        delta_w = 0.00001

        x, y, z = f(sess, scale, offset, targetOrbit=planet)

        ## Plotly equivalent?
        plotlyTrace1 = go.Surface(x=x, # Passing x and y with z, gives the correct axis scaling/values
                                  y=y,
                                  z=z,
                                  showscale=False, # This turns off the scale colormap on the side - don't think we need it
                                  opacity=0.9,
                                  name="Visualization")

        plotlyLayout = go.Layout(title=planet.upper(),
                                 colorway=colorscale,
                                 titlefont=dict(
                                     size=64,        # Quite large
                                     color='#FF1010' # A rather bold red
                                     ),
                                 autosize=True,
                                 yaxis=dict(
                                     autorange = False,
                                     range=[-1, 1]
                                 ),
                                 xaxis=dict(
                                     autorange = False,
                                     range=[-1,1]
                                 ),
                                 margin=dict(
                                     l=65,
                                     r=50,
                                     b=65,
                                     t=200,
                                 ),
        )
    
    
        # Add planet trajectories
        # TODO color planets individually
        planets_phi = phi2(sess, train_X)
        planetTraces = []
        step = int(train_X.shape[0]/8)
        print(step)
        for p in range(8):
            trace = go.Scatter3d(
                x=train_X[step*p:step*(p+1),0],
                y=train_X[step*p:step*(p+1),1],
                z=planets_phi.ravel()[step*p:step*(p+1)], # In order to get from 2d to 1d, use ravel()
                mode='markers',
                name=planets[p].capitalize()
                # marker=dict(
                #     color='rgb(127, 127, 127)',
                #     size=12,
                #     symbol='circle',
                #     line=dict(
                #         color='rgb(204, 204, 204)',
                #         width=1
                #     ),
                #     opacity=0.9
                # )
            )

            planetTraces.append(trace)

        plotlyTrace2 = go.Scatter3d(
            x=train_X[:,0],
            y=train_X[:,1],
            z=planets_phi.ravel(),  # In order to get from 2d to 1d, use ravel()
            mode='markers',
        )

        planets_phi = phi2(sess, train_X)

        #planet_ID = np.repeat(range(8), step)
        planet_ID = list(itertools.chain.from_iterable(itertools.repeat(x, step) for x in planets[:-1]))

        x, y, z = f2(planet_ID, train_X[:, 0], train_X[:, 1], train_X[:, 2], train_X[:, 3], phi2(sess, train_X).ravel(), planet)
        curveFit = go.Surface(x=x,  # Passing x and y with z, gives the correct axis scaling/values
                            y=y,
                            z=z,
                            showscale=False,  # This turns off the scale colormap on the side - don't think we need it
                            opacity=0.9)

        # planetTraces.append(plotlyTrace1)
        planetTraces.append(curveFit)
        plotlyData = planetTraces
        plotlyFig = go.Figure(data=plotlyData, layout=plotlyLayout)
        plotly.offline.plot(plotlyFig, filename=planet+'.html')
        # Plotly done
