#!/usr/bin/python3
import math
import plotly                   # RAH 
import plotly.graph_objs as go  # RAH

import numpy as np
import tensorflow as tf
#import matplotlib.pyplot as plt
#from matplotlib.widgets import Slider, Button, RadioButtons
#from mpl_toolkits.mplot3d import Axes3D

from dataLoader import get_data, datasets
import physicsUtils

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
    op = sess.graph.get_tensor_by_name('Phi/dense/Sigmoid:0')
    return sess.run(op, feed_dict=viz_dic)

def phi2(sess, X):
    viz_dic = {'X:0':np.array(X)}
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
        u, v = physicsUtils.getAvgVelocity(x,y)
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
    u, v = getVelocity(scale, offset, x, y, semimajorAxis=targetOrbit)

    # Calculate phi
    z = phi(sess, x,y,u,v)

    return (np.reshape(x,(n,n)),np.reshape(y,(n,n)), np.reshape(z,(n,n)))
    #return (x, y, np.squeeze(z))



with tf.Session(graph=tf.Graph()) as sess:

    # Load the trained model
    new_saver = tf.train.import_meta_graph('./network/500000.meta')
    new_saver.restore(sess, './network/500000')

    # Load planets and scale values
    scale, offset, (train_X, _, _, _, _, _), benchmark = get_data(shuffle=False)

    # For each planet, plot the values to an interactive <planet>.html
    for planet in planets:
        print ("Planet:",planet)
        if planet != 'none':
            w0 = physicsUtils.radius[planet]
        else:
            w0 = None
        delta_w = 0.00001

        x, y, z = f(sess, scale, offset, targetOrbit=w0)

        ## Plotly equivalent?
        plotlyTrace1 = go.Surface(x=x, # Passing x and y with z, gives the correct axis scaling/values
                                  y=y,
                                  z=z,
                                  showscale=False, # This turns off the scale colormap on the side - don't think we need it
                                  opacity=0.9)
        plotlyLayout = go.Layout(title=planet.upper(),
                                 titlefont=dict(
                                     size=64,        # Quite large
                                     color='#FF1010' # A rather bold red
                                     ),
                                 autosize=True,
                                 yaxis=dict(
                                     autorange = False,
                                     range=[-1,1]
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
    
        plotlyTrace2 = go.Scatter3d(
            x=train_X[:,0],
            y=train_X[:,1],
            z=planets_phi.ravel(), # In order to get from 2d to 1d, use ravel()
            mode='markers',
        )
        
    
        plotlyData = [plotlyTrace1,plotlyTrace2]
        plotlyFig = go.Figure(data=plotlyData, layout=plotlyLayout)
        plotly.offline.plot(plotlyFig, filename=planet+'.html')
        ## Plotly done
