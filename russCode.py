import tensorflow as tf
import numpy as np
from dataLoader import get_raw_data, scaleOrbit


traj, F = get_raw_data()

inputDim = 4 #dd
hiddenDim = 5 # nh

# 4 by N 
xx = traj #xx
xxd = F   #xxd

#Scale weights - using max (vector or scalar scale?)
cc  = np.max(np.abs(xx), axis = 1)
cc2 = np.max(np.abs(xxd), axis = 1)
w1 = 0.01 * np.random.randn((inputDim, hiddenDim))
w1 = np.multiply(w1, (1./cc*np.ones((1, hiddenDim))))

w2 = 0.1 * np.random.randn(hiddenDim, 1)


w1probs = 1 / (1 + np.exp(np.matmul(-w1.T, xx))) 

a = (w2*ones(1,T))
a = w2 * np.ones(w2, axis = 0)

temp = (a).*(w1probs.*(1-w1probs))).T*w1.T .* xxd.T
temp1 = sum(temp, 2); 
rowSum = np.sum(temp,axis=1) #
f1_old = sum(temp1 .* temp1)