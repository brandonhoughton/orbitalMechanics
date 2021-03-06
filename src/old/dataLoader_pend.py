import os
import tensorflow as tf
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale

import numpy as np
from scipy.integrate import odeint


J = os.path.join
E = os.path.exists
dataDir = 'trajectories'
datasets = [
    'earth_xy.npz',
    'jupiter_xy.npz',
    'mars_xy.npz',
    'mercury_xy.npz',
    'neptune_xy.npz',
    'saturn_xy.npz',
    'uranus_xy.npz',
    'venus_xy.npz']
RANDOM_SEED = 42


 
def ODE(qp, t_unused):
    return qp[1],   -np.sin(qp[0])
 
def Htrue(q, p):
    return p ** 2 / 2 + (1 - np.cos(q))
 
def simulate(ic, T_values=None, backward=False):
    if T_values is None: T_values = np.linspace(0, 20, 100)
    if backward:
        f = lambda x, t: ODE(-x, t)
    else:
        f = ODE
    return odeint(f, ic, T_values)

# Expects a 4xn vector with the first two components representing position 
# and the next two components representing the first derivative of position
def scale(vector, method='min-max'):
    if (method == 'physical'):
        # Shift position components
        offset = np.append(np.mean(vector[0:2,:],axis=1), np.zeros(2))
        vector = (vector.transpose() - offset).transpose()
        print (offset)

        # Scale each component uniformly
        scale = 1/np.max(np.abs(vector))
        print (scale)

        return scale, offset, np.multiply(vector, scale)
    elif (method == 'min-max'):
        # Shift position components
        offset = np.mean(vector,axis=0)
        scale = 1 / np.max(np.abs(vector - offset), axis=0)
        s1 = (scale[0] + scale[1]) / 2
        scale[0:2] = s1
        s2 = (scale[2] + scale[3]) / 2
        scale[2:] = s2
        
        # Don't offset velocity
        offset[2:] = 0
        vector = vector - offset
        vector = vector * scale

        return scale, offset, vector
    elif (method == 'none'):
        return np.ones(4), np.ones(4), vector
    else:
        raise(Exception("Not implemented"))


def getBenchmark_old(test_X, test_Y, method):
    if method == 'momentum':
        constant_momentum = 2 * test_X  - np.roll(test_X, 1,axis=0)
        baseline_accuracy  = np.mean((test_Y[1:] - constant_momentum[1:,1:])**2, axis = 0)
        return (baseline_accuracy, constant_momentum)
    else:
        raise(Exception("Not implemented"))

def getBenchmark(test_X, test_Y, method):
    if method == 'momentum_mse':
        constant_momentum = 2 * test_X  - np.roll(test_X, 1,axis=0)
        baseline_accuracy  = np.mean((test_Y[1:] - constant_momentum[1:])**2, axis = 0)
        return (baseline_accuracy, constant_momentum)
    else:
        raise(Exception("Not implemented"))

def get_energy():
    for planet in datasets:
        with np.load(J(dataDir, planet)) as data:
            print(planet,data['energy_total'][213])


def get_data(scaleMethod='min-max', benchmarkMethod='momentum_mse', shuffle= True):
    """ Read the specified orbit and shape it for regression"""   

    samples = []
    X_list = []
    F_list = []
    Y_list = []

    # Load the data to predict the next location and velocity
    for planet in datasets:
        with np.load(J(dataDir, planet)) as data:
            traj = np.array(data['traj'], dtype=np.float32)
            force = np.array(data['F'])


            traj = np.reshape(np.reshape(traj, (-1,1), order='F'),(-1,4))
            force = np.reshape(np.reshape(force, (-1,1), order='F'),(-1,4))

            X = np.roll(traj, shift = 1, axis = 0)[1:]
            F = np.roll(force, shift = 1, axis = 0)[1:]
            Y = traj[:-1]
        

        X_list.append(X)
        F_list.append(F)
        Y_list.append(Y)

        samples.append(X.shape[0])


    # Sample uniformly from each planet
    minSamples = min(samples)

    X_all = np.empty((0,4), dtype=np.float32)
    F_all = np.empty((0,4), dtype=np.float32)
    Y_all = np.empty((0,4), dtype=np.float32)
    for n, x, f, y in zip(samples, X_list, F_list, Y_list):
        #Select a random uniform subset of samples for each planet
        if n > minSamples:
            ids = np.random.choice(range(n), minSamples, replace=False)
            x = x[ids]
            f = f[ids]
            y = y[ids]
        
        print(x.shape)
        X_all = np.append(X_all, x, axis=0)
        F_all = np.append(F_all, f, axis=0)
        Y_all = np.append(Y_all, y, axis=0)


    # Scale Data
    scale, offset, X_all = scaleOrbit(X_all, method=scaleMethod)

    print ('Scale {}, Offset {}, Data{}'.format(scale, offset, X_all.shape))
    print (X_all)

    F_all = scale * F_all

    (train_X, test_X, train_F, test_F, train_Y, test_Y) = train_test_split(X_all, F_all, Y_all, test_size=0, random_state=RANDOM_SEED, shuffle=shuffle)

    benchmarkResult = getBenchmark(X, Y, method=benchmarkMethod)

    return scale, offset, (train_X, test_X, train_F, test_F, train_Y, test_Y), benchmarkResult


def get_data_(scaleMethod='min-max', benchmarkMethod='momentum_mse', shuffle= True):
    """ Read the specified orbit and shape it for regression"""   

    samples = []
    X_prev = []
    X_list = []
    F_list = []
    Y_list = []

    # Load the data to predict the next location and velocity
    for planet in datasets:
        with np.load(J(dataDir, planet)) as data:
            traj = np.array(data['traj'], dtype=np.float32)
            force = np.array(data['F'])

            traj = np.reshape(np.reshape(traj, (-1,1), order='F'),(-1,4))
            force = np.reshape(np.reshape(force, (-1,1), order='F'),(-1,4))

            Xprev = np.roll(traj, shift = 2, axis = 0)[2:]
            X = np.roll(traj, shift = 1, axis = 0)[1:-1]
            F = np.roll(force, shift = 1, axis = 0)[1:-1]
            Y = traj[:-2]
        
        X_prev.append(Xprev)
        X_list.append(X)
        F_list.append(F)
        Y_list.append(Y)

        samples.append(X.shape[0])


    # Sample uniformly from each planet
    minSamples = min(samples)

    X_pall = np.empty((0,4), dtype=np.float32)
    X_all = np.empty((0,4), dtype=np.float32)
    F_all = np.empty((0,4), dtype=np.float32)
    Y_all = np.empty((0,4), dtype=np.float32)
    for n, xp, x, f, y in zip(samples, X_prev, X_list, F_list, Y_list):
        #Select a random uniform subset of samples for each planet
        if n > minSamples:
            ids = np.random.choice(range(n), minSamples, replace=False)
            xp= xp[ids]
            x = x[ids]
            f = f[ids]
            y = y[ids]
        
        print(x.shape)
        X_pall= np.append(X_pall,xp,axis=0)
        X_all = np.append(X_all, x, axis=0)
        F_all = np.append(F_all, f, axis=0)
        Y_all = np.append(Y_all, y, axis=0)


    # Scale Data
    scale, offset, X_all = scaleOrbit(X_all, method=scaleMethod)

    print ('Scale {}, Offset {}, Data{}'.format(scale, offset, X_all.shape))
    print (X_all)

    X_pall= scale * X_pall
    F_all = scale * F_all
    Y_all = scale * Y_all

    (train_Xp, test_xp, train_X, test_X, train_F, test_F, train_Y, test_Y) = train_test_split(X_pall, X_all, F_all, Y_all, test_size=0, random_state=RANDOM_SEED, shuffle=shuffle)

    benchmarkResult = getBenchmark(X, Y, method=benchmarkMethod)

    return scale, offset, (train_Xp, test_xp, train_X, test_X, train_F, test_F, train_Y, test_Y), benchmarkResult



def get_raw_data():
    """ Read the specified orbit and shape it for regression"""   
    return simulate(None)
        

get_energy()