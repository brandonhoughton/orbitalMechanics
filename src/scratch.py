import os
import numpy as np
from scipy.io import loadmat
from matplotlib import pyplot

files = os.listdir('./data/')
data = loadmat('./data/' + files[0])

keys = list(data.keys())
foo = np.array(data[keys[19]])

plt1 = pyplot.imshow(foo[:,:,0])
pyplot.show()
plt2 = pyplot.imshow(0.5 * (foo[:,:,0] - foo[:,:,1]) + 0.5 * (foo[:,:,1] - foo[:,:,2]))
pyplot.show()
