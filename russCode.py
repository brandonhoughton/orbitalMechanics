import tensorflow as tf
import numpy as np
from dataLoader import get_russ_data, scaleOrbit


traj, f = get_russ_data()

######################
# Network Parameters #
######################

inputDim = 4 #dd
hiddenDim = 5 # nh
outDim = 1 #od
lr = 0.01

######################

# 4 by N 
xx = np.array(traj) #xx
xxd = np.array(f)   #xxd
numExamples = xx.shape[0]

#Scale weights - using max (vector scale?)
cc  = np.max(np.abs(xx), axis = 0)
cc2 = np.max(np.abs(xxd), axis = 0)

w1 = 0.01 * np.random.randn(inputDim, hiddenDim)
w1 = w1 * (1/cc*np.ones((hiddenDim,4))).T

w2 = 0.1 * np.random.randn(hiddenDim, outDim)

print(w1)
print(w2)


# Set up computation graph
W1 = tf.Variable(w1, dtype=tf.float32)
B1 = tf.Variable(np.zeros(hiddenDim, np.float32))
W2 = tf.Variable(w2, dtype=tf.float32)
B2 = tf.Variable(np.zeros(outDim, np.float32))

#X = tf.placeholder(tf.float32, shape=(None,4))
#F = tf.placeholder(tf.float32, shape=(None,4))

X = tf.constant(traj, dtype='float')
F = tf.constant(f, dtype='float')


logits = tf.matmul(X, W1) + B1
logits3 = tf.matmul(logits, W2) + B2
cost = tf.reduce_mean(tf.square(logits3))



# Setup gradients
grad_W1, grad_B1, grad_W2, grad_B2 = tf.gradients(cost, [W1, B1, W2, B2])

new_W1 = W1.assign(W1 - lr * grad_W1)
new_B1 = B1.assign(B1 - lr * grad_B1)
new_W2 = W2.assign(W2 - lr * grad_W2)
new_B2 = B2.assign(B2 - lr * grad_B2)

#####################
train_epoch = 5000
display_step = 100
#####################

# Training step
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    # Training cycle
    for epoch in range(train_epoch):
        # Fit training using batch data
        sess.run([new_W1, new_B1, new_W2, new_B2])
        
        # # Compute average loss
        # avg_cost += c / total_batch
        # Display logs per epoch step
        if (epoch+1) % display_step == 0:
            print(sess.run([W1, grad_W1]))

# Energy function calculation
w1probs = 1 / (1 + np.exp(np.matmul(-w1.T, xx.T))) 
a = np.matmul(w2,np.ones((1,numExamples)))
b = (a * (w1probs * (1-w1probs))).T 
temp = np.matmul(b, w1.T) * xxd
temp1 = sum(temp, 2)
rowSum = np.sum(temp,axis=1) #
f1 = sum(temp1 * temp1)

E = np.matmul(w2.T, w1probs)  

print(f1)
print(E)