#!/usr/bin/python3
import plotly  # RAH
import plotly.graph_objs as go  # RAH
import itertools

import numpy as np
import tensorflow as tf
# import matplotlib.pyplot as plt
# from matplotlib.widgets import Slider, Button, RadioButtons
# from mpl_toolkits.mplot3d import Axes3D

import dataLoader

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


# Returns the network evaluated at each point
def phi(sess, x, y, u, v):
    viz_dic = {'X:0': np.array([x, y, u, v]).T}
    op = sess.graph.get_tensor_by_name('Phi/dense/Sigmoid:0')
    return sess.run(op, feed_dict=viz_dic)


def phi_(sess, x, y, u, v, xp, yp, up, vp):
    viz_dic = {'X:0': np.array([x, y, u, v]).T}
    viz_dic['Xp:0'] = np.array([xp, yp, up, vp]).T
    op = sess.graph.get_tensor_by_name('Phi/dense/Sigmoid:0')
    return sess.run(op, feed_dict=viz_dic)


def phi2(sess, X):
    viz_dic = {'X:0': np.array(X)}
    op = sess.graph.get_tensor_by_name('Phi/dense/Sigmoid:0')
    # op = sess.graph.get_tensor_by_name('pred_loss/dense/Sigmoid:0')
    return sess.run(op, feed_dict=viz_dic)

def f2(offset, scale, planetID, planetX, planetY, planetvX, planetvY, planetPhi, targetOrbit=None):
    planetID = np.reshape(planetID, (None))
    planetX = np.reshape(planetX, (None))  # + (offset[0] * scale[0])
    planetY = np.reshape(planetY, (None))  # + (offset[1] * scale[1])
    planetvX = np.reshape(planetvX, (None))  # + (offset[2] * scale[2])
    planetvY = np.reshape(planetvY, (None))  # + (offset[3] * scale[3])

    # print(np.mean(planetX))
    # print(np.mean(planetX - offset[0]))
    # print(np.mean(planetX - (offset[0] * scale[0])))

    planetPhi = np.reshape(planetPhi, (None))

    m1 = physicsUtils.mass_sun
    m2 = np.array([physicsUtils.mass[id] for id in planet_ID])

    # Fit 1/r function
    planetR = np.sqrt(planetX ** 2 + planetY ** 2)

    def hyper(r, A, B):
        return 1 / (A * r) + B

    # Hamiltonian
    def h(x, y, vx, vy):
        return (1 / 2) * (vx ** 2 + vy ** 2) - G * m1 / np.sqrt(x ** 2 + y ** 2)

    # Angular Momentum
    def l(x, y, vx, vy):
        return x * vy - y * vx

    hamiltonian = h(planetX, planetY, planetvX, planetvY)

    momentum = l(planetX, planetY, planetvX, planetvY)

    indep = np.stack([hamiltonian, momentum], axis=-1).T

    def liniar(x, a, b, c):
        return a + b * x[0] + c * x[1]

    popt, pconv = curve_fit(liniar, indep, planetPhi)

    print("a + bx + cy:", popt)
    # print("Covariance: ", pconv)
    # print("Avg: ", sum(pconv)/len(pconv))

    # RMS
    phi_est = liniar(indep, popt[0], popt[1], popt[2])
    sqr_err = np.square(phi_est - planets_phi)
    print('RMS: ', np.sqrt(np.mean(sqr_err)))

    # Per planet RMS
    step = int(planetX.shape[0] / 8)
    for p in range(8):
        print('Planet ', planets[p])
        print(planets[p], 'RMS', np.sqrt(np.mean(sqr_err[p * step:p * step + step])))

    n = 200

    if targetOrbit is not None and targetOrbit != 'none':
        radius = physicsUtils.radius[targetOrbit]
    else:
        radius = None

    # Setup surface plot
    x, y = getMeshGrid(n)
    u, v = getVelocity(scale, offset, x, y, semimajorAxis=radius)

    # if (targetOrbit is None):
    #     avg_m1 = sum(physicsUtils.mass.values())/len(physicsUtils.mass)
    # else:
    #     avg_m1 = physicsUtils.mass[targetOrbit]
    hamiltonian = h(x, y, u, v)

    momentum = l(x, y, u, v)

    indep_grid = np.stack([hamiltonian, momentum], axis=-1).T

    # r = np.sqrt(x ** 2 + y ** 2)

    # Calculate z
    z = liniar(indep_grid, popt[0], popt[1], popt[2]).T

    # Don't messup autoscaling
    # z[z > 0.8] = np.nan

    return (np.reshape(x, (n, n)), np.reshape(y, (n, n)), np.reshape(z, (n, n)))


# Returns the MSE for a linear model fit for phi(x)
def fit_linear(offset, scale, planetX, planetY, planetvX, planetvY, planetPhi, targetOrbit=None):
    # planetID = np.reshape(planetID, (None))
    planetX = np.reshape(planetX, (None)) * (scale[0])
    planetY = np.reshape(planetY, (None)) * (scale[1])
    planetvY = np.reshape(planetvY, (None)) * (scale[3])
    planetvX = np.reshape(planetvX, (None)) * (scale[2])

    planetPhi = np.reshape(planetPhi, (None))

    m1 = physicsUtils.mass_sun

    # Hamiltonian
    def h(x, y, vx, vy):
        return (1 / 2) * (vx ** 2 + vy ** 2) - G * m1 / np.sqrt(x ** 2 + y ** 2)

    # Angular Momentum
    # Pretend the mass is the mass of the sun to scale momentum term properly
    def l(x, y, vx, vy):
        return m1 * (x * vy - y * vx)

    hamiltonian = h(planetX, planetY, planetvX, planetvY)
    print(hamiltonian)

    momentum = l(planetX, planetY, planetvX, planetvY)
    print(momentum)

    indep = np.stack([hamiltonian, momentum], axis=-1).T

    def liniar(x, a, b, c):
        return a + b * x[0] + c * x[1]

    popt, pconv = curve_fit(liniar, indep, planetPhi)

    print("\ta + bx + cy:", popt)
    # print("Covariance: ", pconv)
    # print("Avg: ", sum(pconv)/len(pconv))

    # RMS
    phi_est = liniar(indep, popt[0], popt[1], popt[2])
    sqr_err = np.square(phi_est - planetPhi)
    rms = np.sqrt(np.mean(sqr_err))

    # # Per planet RMS
    # step = int(planetX.shape[0] / 8)
    # for p in range(8):
    #     print('Planet ', planets[p])
    #     print(planets[p], 'RMS', np.sqrt(np.mean(sqr_err[p * step:p * step + step])))

    std = max(np.nanstd(planetPhi), 1e-12)

    print('\tRMS: ', rms, 'STD', std, 'Ratio: ', rms / std)

    return rms, rms / std


if __name__ == "__main__":
    with tf.Session(graph=tf.Graph()) as sess:

        # Load the trained model
        new_saver = tf.train.import_meta_graph('network/400000.meta')
        new_saver.restore(sess, 'network/400000')

        # Load data
        sequence = dataLoader.pendulum.parallel_iterator(batch_size=32, sequence_len=50, num_rep=1)

        ic = sequence[0]  # Initial state of pend
        X = sequence[1]   # Position of pendulum at
        F = sequence[2]

        # scale, offset, (train_Xp, _, train_X, _, _, _, _, _), benchmark = get_data_(shuffle=False)

        # For each planet, plot the values to an interactive <planet>.html
        for planet in planets:
            print("Planet:", planet)
            if planet != 'none':
                w0 = physicsUtils.radius[planet]
                continue
            else:
                w0 = None
            delta_w = 0.00001

            x, y, z = f(sess, scale, offset, targetOrbit=planet)

            ## Plotly equivalent?
            # plotlyTrace1 = go.Surface(x=x, # Passing x and y with z, gives the correct axis scaling/values
            #                           y=y,
            #                           z=z,
            #                           showscale=False, # This turns off the scale colormap on the side - don't think we need it
            #                           opacity=0.9,
            #                           name="Visualization")
            plotlyTrace1 = go.Scatter(x=np.sqrt(x ** 2 + y ** 2),
                                      # Passing x and y with z, gives the correct axis scaling/values
                                      y=z,
                                      opacity=0.9,
                                      name="Visualization")

            plotlyLayout = go.Layout(title=planet.upper(),
                                     colorway=colorscale,
                                     titlefont=dict(
                                         size=64,  # Quite large
                                         color='#FF1010'  # A rather bold red
                                     ),
                                     scene=dict(
                                         yaxis=dict(
                                             autorange=False,
                                             range=[-1, 1]
                                         ),
                                         xaxis=dict(
                                             autorange=False,
                                             range=[-1, 1]
                                         ),
                                         zaxis=dict(
                                             autorange=False,
                                             range=[0, 0.8]
                                         )),
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
            step = int(train_X.shape[0] / 8)
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
                    # x=np.log(
                    #     np.sqrt(train_X[step*p:step*(p+1),0] ** 2 + train_X[step*p:step*(p+1),1] ** 2)),
                    # y=np.log(
                    #     planets_phi.ravel()[step*p:step*(p+1)]), # In order to get from 2d to 1d, use ravel()
                    x=np.sqrt(train_X[step * p:step * (p + 1), 0] ** 2 + train_X[step * p:step * (p + 1), 1] ** 2),
                    y=planets_phi.ravel()[step * p:step * (p + 1)],  # In order to get from 2d to 1d, use ravel()
                    mode='markers',
                    name=planets[p].capitalize()
                )

                planetTraces.append(trace)

            # plotlyTrace2 = go.Scatter3d(
            #     x=train_X[:,0],
            #     y=train_X[:,1],
            #     z=planets_phi.ravel(),  # In order to get from 2d to 1d, use ravel()
            #     mode='markers',
            # )

            planets_phi = phi2(sess, train_X)

            # planet_ID = np.repeat(range(8), step)
            planet_ID = list(itertools.chain.from_iterable(itertools.repeat(x, step) for x in planets[:-1]))

            x, y, z = f2(offset, scale, planet_ID, train_X[:, 0], train_X[:, 1], train_X[:, 2], train_X[:, 3],
                         phi2(sess, train_X).ravel(), planet)
            # curveFit = go.Surface(x=x,  # Passing x and y with z, gives the correct axis scaling/values
            #                     y=y,
            #                     z=z,
            #                     showscale=False,  # This turns off the scale colormap on the side - don't think we need it
            #                     opacity=0.9)
            curveFit = go.Scatter(x=np.sqrt(x ** 2 + y ** 2).ravel(),
                                  # Passing x and y with z, gives the correct axis scaling/values
                                  y=z,
                                  opacity=0.9)

            # planetTraces.append(plotlyTrace1)
            planetTraces.append(curveFit)
            plotlyData = planetTraces
            plotlyFig = go.Figure(data=plotlyData, layout=plotlyLayout)
            plotly.offline.plot(plotlyFig, filename=planet + '.html')
            # Plotly done
