import os
import random
import tensorflow as tf
import numpy as np
import datetime

from dataLoader import get_data

def variable_summaries(var):
  """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
  with tf.name_scope('summaries'):
    mean = tf.reduce_mean(var)
    tf.summary.scalar('mean', mean)
    with tf.name_scope('stddev'):
      stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
    tf.summary.scalar('stddev', stddev)
    tf.summary.scalar('max', tf.reduce_max(var))
    tf.summary.scalar('min', tf.reduce_min(var))
    tf.summary.histogram('histogram', var)

def singleLayer(X, outDim = 50):
    #Hidden layers
    head = tf.layers.dense(X, outDim, activation=tf.nn.sigmoid, use_bias=True)
    return head

def trippleLayer(X, outDim = 16):
    head = tf.layers.dense(X, 64, activation=tf.nn.sigmoid, use_bias=True)
    head = tf.layers.dense(head, 24, activation=tf.nn.sigmoid, use_bias=True)
    head = tf.layers.dense(head, outDim, activation=tf.nn.sigmoid, use_bias=True)
    return head

def setupLoss(baseNetwork, X, F, Y):

    # Network output
    Phi = tf.layers.dense(baseNetwork, 1, activation=tf.nn.sigmoid, use_bias=True)
    Auto = tf.layers.dense(baseNetwork, 4, activation=tf.nn.sigmoid, use_bias=True)
    
    deltaPhi = tf.gradients(Phi, [X])[0]

    # Cosine distance between F and X
    dotProd = tf.reduce_sum(tf.multiply(F, deltaPhi)/tf.expand_dims(tf.norm(F, axis=1)*tf.norm(deltaPhi, axis=1), dim=1), axis=1)

    # Mean Squared 
    gradTerm = tf.reduce_mean(tf.norm(deltaPhi, axis=1) - 1)

    gradMag = tf.norm(deltaPhi, axis=1)

    loss = tf.abs(dotProd)

    return loss, [Phi, deltaPhi, dotProd, gradTerm, autoTerm]


#####################
train_epoch = 100000
display_step = 1000000
summary_step = 100
#####################
    
def main():

    # Load data
    scale, offset, (train_X, test_X, train_F, test_F, train_Y, test_Y), benchmark, planet = get_data()

    # Load data onto GPU memory - ensure network layers have GPU support
    with tf.device('/gpu:0'):

        # Define GPU constants
        X = tf.identity(tf.constant(train_X, dtype= tf.float32))
        F = tf.identity(tf.constant(train_F, dtype= tf.float32))
        Y = tf.identity(tf.constant(train_Y, dtype= tf.float32))

        ## Define network
        with tf.name_scope('Base_Network'):
            baseNetwork = singleLayer(X)

        with tf.name_scope('Phi'):
            Phi = singleLayer(baseNetwork, outDim = 1)
            variable_summaries(Phi)

        with tf.name_scope('Prediction'):
            Pred = singleLayer(baseNetwork, outDim = 4)
            variable_summaries(Pred)

        ## Define Loss
        with tf.name_scope('gradPhi'):
            gradPhi = tf.gradients(Phi, [X])[0]
            variable_summaries(gradPhi)

        # Calculate dot product 
        with tf.name_scope('dotProd'):
            dotProd = tf.reduce_sum(tf.multiply(F, gradPhi)/tf.expand_dims(tf.norm(F, axis=1)*tf.norm(gradPhi, axis=1), dim=1), axis=1)
            variable_summaries(dotProd)

        # Calculate gradient regualization term
        with tf.name_scope('gradMag'):
            gradMag = tf.norm(gradPhi, axis=1)
            variable_summaries(gradMag)

        with tf.name_scope('pred_loss'):
            predLoss = tf.losses.huber_loss(Y,Pred)
            variable_summaries(predLoss)

        with tf.name_scope('loss'):
            alpha = tf.constant(0.01, dtype=tf.float32) # Scaling factor for magnitude of gradient
            beta  = tf.constant(0.1, dtype=tf.float32)  # Scaling factor for prediction of next time step 
            loss = tf.reduce_mean(tf.abs(dotProd) + alpha * gradMag + beta * predLoss)
            variable_summaries(loss)

        with tf.name_scope('train'):
            train_step = tf.train.AdamOptimizer().minimize(loss)


        ## Collect summary stats
        merged = tf.summary.merge_all()
        timeStr = datetime.datetime.today().strftime('%Y-%m-%d_%H-%M')
        train_writer = tf.summary.FileWriter('./train/' + planet + '/' + timeStr)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            for epoch in range(train_epoch):
                if (epoch+1) % summary_step == 0:
                    summary = sess.run([merged])[0]
                    train_writer.add_summary(summary, epoch)

                l, _ = sess.run([loss, train_step])


main()