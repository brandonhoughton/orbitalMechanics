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
    viz_dic = {'X:0': np.array([x, y, u, v]).T}
    op = sess.graph.get_tensor_by_name('Phi/dense/Sigmoid:0')
    return sess.run(op, feed_dict=viz_dic)


# def phi_(sess, x, y, u, v, xp, yp, up, vp):
#     viz_dic = {'X:0': np.array([x, y, u, v]).T}
#     viz_dic['Xp:0'] = np.array([x_, y_, u_, v_]).T
#     op = sess.graph.get_tensor_by_name('Phi/dense/Sigmoid:0')
#     return sess.run(op, feed_dict=viz_dic)
#

def phi2(sess, X):
    viz_dic = {'X:0':np.array(X)}
    op = sess.graph.get_tensor_by_name('Phi/dense/Sigmoid:0')
    #op = sess.graph.get_tensor_by_name('pred_loss/dense/Sigmoid:0')
    return sess.run(op, feed_dict=viz_dic)


def getMeshGrid(n):
    x = np.linspace(-1, 1, n)
    y = np.linspace(-1, 1, n)
    x, y = np.meshgrid(x, y)
    return np.reshape(x, (-1)), np.reshape(y, (-1))


def getVelocity(scale, offset, x, y, semimajorAxis = None):
    # offset = np.zeros([4])
    # x, y = (x , y) / np.expand_dims(scale[:2], 1)
    if semimajorAxis is None:
        u, v = physicsUtils.getAvgVelocity(x, y)
    else:
        u, v = physicsUtils.getVelocity(semimajorAxis, x, y)

    return u,v
    # return (u, v) * np.expand_dims(scale[2:4], 1)


# Returns an array of points for phi(x,y)
# Include target orbit to calulate phi for eliptical paths about the sepcified
# semimajor axis
def f(sess, scale, offset, targetOrbit=None):
    n = 200
    # Setup surface plot
    x, y = getMeshGrid(n)

    #z = np.ones_like(x)

    # Setup velocity
    if targetOrbit is not None and targetOrbit != 'none':
        radius = physicsUtils.radius[targetOrbit]
    else:
        radius = None

    u, v = getVelocity(scale, offset, x, y, semimajorAxis=radius)

    # Calculate phi
    z = phi(sess, x,y,u,v)

    return (np.reshape(x,(n,n)),np.reshape(y,(n,n)), np.reshape(z,(n,n)))
    #return (x, y, np.squeeze(z))

def f2(offset, scale, planetID, planetX, planetY, planetvX, planetvY, planetPhi, targetOrbit = None):
    planetID = np.reshape(planetID, (None))
    planetX = np.reshape(planetX, (None))
    planetY = np.reshape(planetY, (None))
    planetvX = np.reshape(planetvX, (None))
    planetvY = np.reshape(planetvY, (None))

    planetPhi = np.reshape(planetPhi, (None))

    #Check velocity
    # stack = np.stack([planetX, planetY, planetvX, planetvY], axis=1)[45:50]

    # u_hat, v_hat = getVelocity(scale, offset, stack[0,:], stack[1,:])
    # print(stack[0,:], u_hat, ", ", stack[1,:], v_hat)

    m1 = physicsUtils.mass_sun
    m2 = np.array([physicsUtils.mass[id] for id in planet_ID])

    # Fit 1/r function
    planetR = np.sqrt(planetX ** 2 + planetY ** 2)
    def hyper(r, A, B):
        return 1/(A*r) + B

    # Hamiltonian
    def h(x, y, vx, vy):
        return (1 / 2) * (vx ** 2 + vy ** 2) - G * m1 / np.sqrt(x ** 2 + y ** 2)

    # Angular Momentum
    def l(x, y, vx, vy):
        return x * vy - y * vx

    hamiltonian = h(planetX / scale[0] + offset[0],
                    planetY / scale[1] + offset[1],
                    planetvX / scale[2] + offset[2],
                    planetvY / scale[3] + offset[3])

    momentum = l(planetX / scale[0] + offset[0],
                 planetY / scale[1] + offset[1],
                 planetvX / scale[2] + offset[2],
                 planetvY / scale[3] + offset[3])

    indep = np.stack([hamiltonian, momentum], axis=-1).T

    def linear(x, a, b, c, d, e):
        return a + b*(x[0]+d) + c*(x[1]+e)

    popt, pconv = curve_fit(linear, indep, planetPhi)

    print("a + bx + cy:", popt)
    # print("Covariance: ", pconv)
    # print("Avg: ", sum(pconv)/len(pconv))

    # RMS
    phi_est = linear(indep, popt[0], popt[1], popt[2], popt[3], popt[4])
    sqr_err = np.square(phi_est - planets_phi)
    print('RMS: ', np.sqrt(np.mean(sqr_err)))

    # Per planet RMS
    step = int(planetX.shape[0] / 8)
    for p in range(8):
        print('Planet ', planets[p])
        print(planets[p], 'RMS', np.sqrt(np.mean(sqr_err[p*step:p*step + step])))


    n = 200

    if targetOrbit is not None and targetOrbit != 'none':
        radius = physicsUtils.radius[targetOrbit]
    else:
        radius = None

    # Setup surface plot
    x, y = getMeshGrid(n)
    u, v = getVelocity(scale, offset, x / scale[0] + offset[0], y / scale[1] + offset[0], semimajorAxis=radius)

    # if (targetOrbit is None):
    #     avg_m1 = sum(physicsUtils.mass.values())/len(physicsUtils.mass)
    # else:
    #     avg_m1 = physicsUtils.mass[targetOrbit]
    hamiltonian = h(x, y, u, v)

    momentum = l(x, y, u, v)

    indep_grid = np.stack([hamiltonian, momentum], axis=-1).T

    #r = np.sqrt(x ** 2 + y ** 2)

    # Calculate z
    # z = linear(indep_grid, popt[0], popt[1], popt[2]).T
    z = linear(indep, popt[0], popt[1], popt[2], popt[3], popt[4]).T

    # Don't messup autoscaling
    # z[z > 0.8] = np.nan

    # return (np.reshape(x,(n,n)),np.reshape(y,(n,n)), np.reshape(z,(n,n)))
    return (planetX / scale[0] + offset[0], planetY / scale[1] + offset[1], np.reshape(z, (None)))





with tf.Session(graph=tf.Graph()) as sess:

    # Load the trained model
    new_saver = tf.train.import_meta_graph('network/400000.meta')
    new_saver.restore(sess, 'network/400000')


    # Load planets and scale values
    # scale, offset, (train_x, _, _, _, _, _), benchmark = get_data(shuffle=False, scaleMethod='no_scale')
    scale, offset, (train_X, _, _, _, _, _), benchmark = get_data(shuffle=False)
    #scale, offset, (train_Xp, _, train_X, _, _, _, _, _), benchmark = get_data_(shuffle=False)

    # For each planet, plot the values to an interactive <planet>.html
    for planet in planets:
        print ("Planet:",planet)
        if planet != 'none':
            w0 = physicsUtils.radius[planet]
            continue
        else:
            w0 = None
        delta_w = 0.00001

        # x, y, z = f(sess, scale, offset, targetOrbit=planet)
        #
        # ## Plotly equivalent?
        # # plotlyTrace1 = go.Surface(x=x, # Passing x and y with z, gives the correct axis scaling/values
        # #                           y=y,
        # #                           z=z,
        # #                           showscale=False, # This turns off the scale colormap on the side - don't think we need it
        # #                           opacity=0.9,
        # #                           name="Visualization")
        # plotlyTrace1 = go.Scatter(x=np.sqrt(x ** 2 + y ** 2), # Passing x and y with z, gives the correct axis scaling/values
        #                           y=z,
        #                           opacity=0.9,
        #                           name="Visualization")

        plotlyLayout = go.Layout(title='Learned Invariants', #title=planet.upper(),
                                 colorway=colorscale,
                                 titlefont=dict(
                                     size=64,        # Quite large
                                     color='#433f3f' # A rather bold red
                                     ),
                                 xaxis=dict(
                                     title='Learned Phi',
                                     titlefont=dict(
                                         size=32,
                                         color='#444444'
                                     )
                                 ),
                                 yaxis=dict(
                                     title='Radius',
                                     titlefont=dict(
                                         size=32,
                                         color='#444444'
                                     )
                                 ),
                                 legend=dict(font=dict(size=18)),
                                # scene=dict(
                                #      yaxis=dict(
                                #          autorange = False,
                                #          range=[-1, 1]
                                #      ),
                                #      xaxis=dict(
                                #          autorange = False,
                                #          range=[-1, 1]
                                #      ),
                                #      zaxis=dict(
                                #          autorange=False,
                                #          range=[0, 0.8]
                                #      )),
                                 margin=dict(
                                     l=65,
                                     r=50,
                                     b=65,
                                     t=200,
                                 ),
        )
    
    
        # Add planet trajectories
        planets_phi = phi2(sess, train_X)
        planetTraces = []
        step = int(train_X.shape[0]/8)
        X = train_X / scale + offset
        rad = np.sqrt(X[:, 0] ** 2 + X[:, 1] ** 2)

        print(step)
        for p in range(8):
            # trace = go.Scatter3d(
            #     x=train_X[step*p:step*(p+1),0],
            #     y=train_X[step*p:step*(p+1),1],
            #     z=planets_phi.ravel()[step*p:step*(p+1)], # In order to get from 2d to 1d, use ravel()
            #     mode='markers',
            #     name=planets[p].capitalize()
            # )
            trace = go.Scatter(
                x=rad[step * p:step * (p + 1)],
                y=planets_phi.ravel()[step*p:step*(p+1)], # In order to get from 2d to 1d, use ravel()
                # x=np.log(
                #     rad[step * p:step * (p + 1)]),
                # y=np.log(
                #     planets_phi.ravel()[step * p:step * (p + 1)]),  # In order to get from 2d to 1d, use ravel()
                mode='markers',
                marker=dict(
                    size=11,
                ),
                name=planets[p].capitalize()
            )

            planetTraces.append(trace)

        # plotlyTrace2 = go.Scatter3d(
        #     x=train_X[:,0],
        #     y=train_X[:,1],
        #     z=planets_phi.ravel(),  # In order to get from 2d to 1d, use ravel()
        #     mode='markers',
        # )

        #planet_ID = np.repeat(range(8), step)
        planet_ID = list(itertools.chain.from_iterable(itertools.repeat(x, step) for x in planets[:-1]))

        x, y, z = f2(offset, scale, planet_ID, train_X[:, 0], train_X[:, 1], train_X[:, 2], train_X[:, 3], phi2(sess, train_X).ravel(), planet)
        # curveFit = go.Surface(x=x,  # Passing x and y with z, gives the correct axis scaling/values
        #                     y=y,
        #                     z=z,
        #                     showscale=False,  # This turns off the scale colormap on the side - don't think we need it
        #                     opacity=0.9)
        curveFit = go.Scatter(
            x=np.sqrt(x ** 2 + y ** 2).ravel(), # Passing x and y with z, gives the correct axis scaling/values
            y=z,
            # x=np.log(np.sqrt(x ** 2 + y ** 2)).ravel(),  # Passing x and y with z, gives the correct axis scaling/values
            # y=np.log(z),
            marker=dict(
              size=11,
            ),
            opacity=0.9,
              mode='markers',
              name="Linear Fit")

        # planetTraces.append(plotlyTrace1)
        planetTraces.append(curveFit)
        plotlyData = planetTraces
        plotlyFig = go.Figure(data=plotlyData, layout=plotlyLayout)
        plotly.offline.plot(plotlyFig, filename=planet+'.html')
        # Plotly done
