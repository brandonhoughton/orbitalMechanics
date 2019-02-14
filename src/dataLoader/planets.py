import math
import os
import numpy as np

from sklearn.model_selection import train_test_split

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

# Expects a 4xn vector with the first two components representing position 
# and the next two components representing the first derivative of position
def scaleOrbit(vector, method='min-max'):
    if (method == 'physical'):
        # # Shift position components
        # offset = np.append(np.mean(vector[0:2,:],axis=1), np.zeros(2))
        # vector = (vector.transpose() - offset).transpose()
        # print (offset)

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
    elif (method == 'no_scale'):
        offset = np.zeros([4])
        scale = 1 / np.max(np.abs(vector - offset), axis=0)
        s1 = (scale[0] + scale[1]) / 2
        scale[0:2] = s1
        s2 = (scale[2] + scale[3]) / 2
        scale[2:] = s2

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


def get_data_(scaleMethod='no_scale', benchmarkMethod='momentum_mse', shuffle= True):
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
        
        # print(x.shape)
        X_pall= np.append(X_pall,xp,axis=0)
        X_all = np.append(X_all, x, axis=0)
        F_all = np.append(F_all, f, axis=0)
        Y_all = np.append(Y_all, y, axis=0)


    # Scale Data
    scale, offset, X_all = scaleOrbit(X_all, method=scaleMethod)
    #
    # print ('Scale {}, Offset {}, Data{}'.format(scale, offset, X_all.shape))
    # print (X_all)

    X_pall= scale * X_pall
    F_all = scale * F_all
    Y_all = scale * Y_all

    (train_Xp, test_xp, train_X, test_X, train_F, test_F, train_Y, test_Y) = train_test_split(X_pall, X_all, F_all, Y_all, test_size=0, random_state=RANDOM_SEED, shuffle=shuffle)

    benchmarkResult = getBenchmark(X, Y, method=benchmarkMethod)

    return scale, offset, (train_Xp, test_xp, train_X, test_X, train_F, test_F, train_Y, test_Y), benchmarkResult


def get_data_segmented(scaleMethod='no_scale', benchmarkMethod='momentum_mse', shuffle=True, seed=None):
    """ Read the specified orbit and shape it for regression"""
    X_prev = []
    X_list = []
    F_list = []
    Y_list = []

    # Load the data to predict the next location and velocity
    for planet in datasets:
        with np.load(J(dataDir, planet)) as data:
            traj = np.array(data['traj'], dtype=np.float32)
            force = np.array(data['F'])

            traj = np.reshape(np.reshape(traj, (-1, 1), order='F'), (-1, 4))
            force = np.reshape(np.reshape(force, (-1, 1), order='F'), (-1, 4))

            Xprev = np.roll(traj, shift=2, axis=0)[2:]
            X = np.roll(traj, shift=1, axis=0)[1:-1]
            F = np.roll(force, shift=1, axis=0)[1:-1]
            Y = traj[:-2]

        X_prev.append(Xprev)
        X_list.append(X)
        F_list.append(F)
        Y_list.append(Y)

    # Sample uniformly from each planet
    X_pall = np.empty((0, 4), dtype=np.float32)
    X_all = np.empty((0, 4), dtype=np.float32)
    F_all = np.empty((0, 4), dtype=np.float32)
    Y_all = np.empty((0, 4), dtype=np.float32)

    np.random.seed(seed)

    # Remove one quadrant from the data
    def filter_section(zipped):
        xp, x, f, y = zipped
        theta = np.random.random() * 2 * math.pi
        theta_max = (theta + 0.5 * math.pi) % (2 * math.pi)

        def between(arr):
            _, pt, _, _ = arr
            x, y, _, _ = pt
            angle = math.atan2(y, x) % (2 * math.pi)
            if (theta < theta_max):
                return not (theta < angle and angle < theta_max)
            elif (theta > theta_max):
                return not ( theta < angle or angle < theta_max)
            return True


        xp, x, f, y = zip(*filter(between, zip(xp, x, f, y)))
        return np.array(xp), np.array(x), np.array(f), np.array(y)

    data = [filter_section(group) for group in zip(X_prev, X_list, F_list, Y_list)]
    minSamples = min([x.shape[0] for _, x, _, _ in data])

    for xp, x, f, y in data:
        # Select a random uniform subset of samples for each planet
        if len(x) > minSamples:
            ids = np.random.choice(range(len(x)), minSamples, replace=False)
            xp = xp[ids]
            x = x[ids]
            f = f[ids]
            y = y[ids]

        print(x.shape)
        X_pall = np.append(X_pall, xp, axis=0)
        X_all = np.append(X_all, x, axis=0)
        F_all = np.append(F_all, f, axis=0)
        Y_all = np.append(Y_all, y, axis=0)

    # Scale Data
    scale, offset, X_all = scaleOrbit(X_all, method=scaleMethod)

    print('Scale {}, Offset {}, Data{}'.format(scale, offset, X_all.shape))
    print(X_all)

    X_pall = scale * X_pall
    F_all = scale * F_all
    Y_all = scale * Y_all

    (train_Xp, test_xp, train_X, test_X, train_F, test_F, train_Y, test_Y) = train_test_split(X_pall, X_all, F_all,
                                                                                              Y_all, test_size=0,
                                                                                              random_state=RANDOM_SEED,
                                                                                              shuffle=shuffle)

    benchmarkResult = getBenchmark(X, Y, method=benchmarkMethod)

    return scale, offset, (train_Xp, test_xp, train_X, test_X, train_F, test_F, train_Y, test_Y), benchmarkResult



def get_russ_data(planet = 0, scaleMethod='min-max', benchmarkMethod='momentum', shuffle= True):
    """ Read the specified orbit and shape it for regression"""   

    X = []
    Y = []

    # Load the data to predict the next location and velocity
    with np.load(J(dataDir,datasets[planet])) as data:
        traj = data['traj']
        f = data['F']


        traj = np.reshape(np.reshape(traj, (-1,1), order='F'),(-1,4))
        f = np.reshape(np.reshape(f, (-1,1), order='F'),(-1,4))

        print(traj)
        print(f)


        # X = np.roll(traj, shift = 1, axis = 0)[1:]
        # X = np.insert(X, 0, 1, axis=1)
        # Y = traj[1:]

    return traj, f, datasets[planet]


def get_raw_data(planet = 0, predictionHoizon = 1):
    """ Read the specified orbit and shape it for regression"""   

    X = []
    Y = []

    # Load the data to predict the next location and velocity
    with np.load(J(dataDir,datasets[planet])) as data:
        traj = data['traj']

       

        traj = np.reshape(np.reshape(traj, (-1,1), order='F'),(-1,4))

        X = np.roll(traj, shift = predictionHoizon, axis = 0)[predictionHoizon:]
        Y = traj[predictionHoizon:]

        return X, Y
        

# get_data_segmented(shuffle=False, seed=42)