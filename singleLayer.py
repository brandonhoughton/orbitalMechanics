# @vinhkhuc vinhkhuc/simple_mlp_tensorflow.py
# Last active 16 days ago •
# Code
# Revisions 5
# Stars 69
# Forks 30
# Simple Feedforward Neural Network using TensorFlow
# simple_mlp_tensorflow.py
# Implementation of a simple MLP network with one hidden layer. Tested on the iris data set.
# Requires: numpy, sklearn>=0.18.1, tensorflow>=1.0

# NOTE: In order to make the code simple, we rewrite x * W_1 + b_1 = x' * W_1'
# where x' = [x | 1] and W_1' is the matrix W_1 appended with a new row with elements b_1's.
# Similarly, for h * W_2 + b_2

import os
import random
import tensorflow as tf
import numpy as np

from sklearn import datasets
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

from dataLoader import get_data

RANDOM_SEED = 42
tf.set_random_seed(RANDOM_SEED)

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

def init_weights(shape):
    """ Weight initialization """
    weights = tf.random_normal(shape, mean= 0, stddev=1)
    return tf.Variable(weights)

def forwardprop(X, w_1, w_2, w_3):
    """
    Forward-propagation.
    IMPORTANT: yhat is not softmax since TensorFlow's softmax_cross_entropy_with_logits() does that internally.
    """
    h    = tf.nn.relu(tf.matmul(X, w_1))  # The \sigma function
    h2    = tf.nn.relu(tf.matmul(h, w_2))  # The \sigma function
    yhat = tf.matmul(h2, w_3)  # The \varphi function
    return yhat

def main():
    scale, offset, (train_X, test_X, train_y, test_y) = get_data()
    print(train_X.shape[0])

    # Layer's sizes
    x_size = train_X.shape[1]   # Number of input nodes: 4 features x, y, dx, dy
    h_size = 64                  # Number of hidden nodes
    h1_size = 32                  # Number of hidden nodes
    y_size = train_y.shape[1]   # Number of outcomes 4 features x, y, dx, dy

    # Symbols
    
    with tf.device('/gpu:0'):
        X = tf.constant(train_X, dtype='float')
        y = tf.constant(train_y, dtype='float')

        X_test = tf.constant(test_X, dtype='float')
        # Y_test = tf.constant(test_y, dtype='float')

        # X = tf.placeholder("float", shape=[None, x_size])
        # y = tf.placeholder("float", shape=[None, y_size])
        lr = tf.placeholder("float", shape=[])

        # Weight initializations
        w_1 = tf.Variable(np.random.randn(x_size, h_size)  * np.sqrt(2/x_size), dtype = 'float') #init_weights((x_size, h_size))
        w_2 = tf.Variable(np.random.randn(h_size, h1_size) * np.sqrt(2/h_size), dtype = 'float') #w_2 = init_weights((h_size, h1_size))
        w_3 = tf.Variable(np.random.randn(h1_size, y_size) * np.sqrt(2/h1_size), dtype = 'float') #w_3 = init_weights((h1_size, y_size))
        
        

        # Forward propagation
        yhat      = forwardprop(X, w_1, w_2, w_3)
        yhat_test  = forwardprop(X_test, w_1, w_2, w_3)

        # Backward propagation
        cost    = tf.reduce_mean(tf.losses.huber_loss(labels=y,predictions=yhat))
        updates = tf.train.AdamOptimizer(learning_rate=lr).minimize(cost)

        # Run SGD
        sess = tf.Session()
        init = tf.global_variables_initializer()
        sess.run(init)

        naive_accuracy  = np.mean((test_y - test_X[:,1:])**2, axis = 0)
        print('Naive Accruacy:', naive_accuracy)

        constant_momentum_1 = 2 * test_X  - np.roll(test_X, 1,axis=0)
        baseline_accuracy  = np.mean((test_y[1:] - constant_momentum_1[1:,1:])**2, axis = 0)
        print('Momentum Accruacy:', baseline_accuracy)


        for epoch in range(100000):
            learning_rate = 0.005 * 0.99**epoch
            learning_rate = max(learning_rate, 0.0005)
            # Train with each example   
            for _ in range(1200):  
                sess.run(updates, feed_dict={lr:learning_rate})

            #train_accuracy = np.mean((train_y - sess.run(yhat))**2, axis = 0)
            test_accuracy  = np.mean((test_y - sess.run(yhat_test))**2, axis = 0)

            train_accuracy = mean_squared_error(train_y, sess.run(yhat))

            

            print("Epoch = %d lr = %f" % (epoch + 1, learning_rate))
            print('Train Accuracy:', train_accuracy)
            print('Test Accuracy:', test_accuracy)
            pos = np.sum((baseline_accuracy / test_accuracy )[0:2])
            vel = np.sum((baseline_accuracy / test_accuracy )[2:])
            print("Test vs Baseline: pos: {:.3%} vel: {:.3%}".format(pos, vel))
            pos = np.sum((baseline_accuracy / train_accuracy )[0:2])
            vel = np.sum((baseline_accuracy / train_accuracy )[2:])
            print("Train vs Baseline: pos: {:.3%} vel: {:.3%}".format(pos, vel))
            

        sess.close()

if __name__ == '__main__':
    main()