import os
import random
import tensorflow as tf
import numpy as np

from dataLoader import get_data

def singleLayer(X, F, Y):
    head = tf.nn.f

def main():


    # Load data
    scale, offset, (train_X, test_X, train_F, test_F, train_Y, test_Y), benchmark = get_data()

    # Define network


    # Layer's sizes
    x_size = train_X.shape[1]   # Number of input nodes: 4 features x, y, dx, dy
    h_size = 256                  # Number of hidden nodes
    h1_size = 32                  # Number of hidden nodes
    y_size = train_y.shape[1]   # Number of outcomes 4 features x, y, dx, dy

    # Load data onto GPU memory
    with tf.device('/gpu:0'):