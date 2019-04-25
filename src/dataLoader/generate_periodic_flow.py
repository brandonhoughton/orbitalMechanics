import time
import numpy as np
import scipy.io
from matplotlib import pyplot as plt


def sin(x, y, t):
    theta_x = x / 20
    theta_y = y / 20
    theta_t = t * np.pi / 180

    return np.sin(theta_x) * np.sin(theta_t) + np.cos(theta_y) * np.cos(theta_t)  # 0
    # return np.sin(theta_x) * np.sin(theta_t) + np.cos(theta_y) * np.cos(theta_t/2)  # 1
    # return (np.sin(theta_x) + np.cos(theta_y)) * np.cos(theta_t)  # 2
    # return np.sin(theta_x + theta_t) + np.cos(theta_y + theta_t / 0.5)  # 3
    # return np.sin(theta_x*2) * np.sin(theta_t * 8) + np.cos(theta_y*2) * np.cos(theta_t * 8 /2)  # 4


def generate():
    # Fixed dimensions equal to that of velocity_and_vorticity_field_1200.mat
    shape = [360, 279, 1000]
    arr = np.fromfunction(sin, shape)
    return arr


def visualize(arr):
    # Visualize the function
    img = None
    for i in range(arr.shape[-1]):
        im = arr[:, :, i]
        if img is None:
            img = plt.imshow(im)
        else:
            img.set_data(im)
        plt.pause(.001)
        plt.draw()

def load():
    return scipy.io.loadmat('data/turbulence/velocity_and_vorticity_field_1200s.mat')['U_t']

def save(arr):
    scipy.io.savemat('../../data/turbulence/periodic_4.mat', {'U_t': arr})
    pass
    # Save file
    # TODO

if __name__ == "__main__":
    arr = generate()
    # arr = load()
    visualize(arr)
    save(arr)
