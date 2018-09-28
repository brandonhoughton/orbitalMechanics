import os
import random
import tensorflow as tf
import numpy as np
import datetime

from dataLoader import get_data

def variable_summaries(var, groupName):
  """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
  with tf.name_scope(groupName):
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


#####################
train_epoch = 500000
display_step =  1000
summary_step =  2000
pre_train_steps = 1000
#####################
a = 0.01 # GradNorm Weight
b = 0.01 # Prediction Weight
g = 0.01 # Scale for Phi
lr = 0.01
#####################
    
def main():

    # Load data
    scale, offset, (train_X, test_X, train_F, test_F, train_Y, test_Y), benchmark = get_data(shuffle=False)

    # Load data onto GPU memory - ensure network layers have GPU support
    #with tf.device('/gpu:0'):
    if True:
        # Define GPU constants
        X = tf.identity(tf.constant(train_X, dtype= tf.float32))
        F = tf.identity(tf.constant(train_F, dtype= tf.float32))
        # Y = tf.identity(tf.constant(train_Y, dtype= tf.float32))

        ## Define network
        with tf.name_scope('Base_Network'):
            baseNetwork = trippleLayer(X)

        with tf.name_scope('Phi'):
            Phi = singleLayer(baseNetwork, outDim = 1)
            
        # with tf.name_scope('Prediction'):
        #     Pred = singleLayer(baseNetwork, outDim = 4)

        ## Define Loss
        with tf.name_scope('gradPhi'):
            gradPhi = tf.gradients(Phi, [X])[0]
            
        # Calculate dot product 
        with tf.name_scope('dotProd'):
            dotProd = tf.reduce_sum(tf.multiply(F, gradPhi)/tf.expand_dims(tf.norm(F, axis=1)*tf.norm(gradPhi, axis=1), dim=1), axis=1)            

        # Calculate gradient regualization term
        with tf.name_scope('gradMag'):
            gradMag = tf.norm(gradPhi, axis=1)   

        with tf.name_scope('grad_loss'):
            gradLoss = tf.reduce_mean(tf.square(gradMag - 1))       

        # with tf.name_scope('pred_loss'):
        #     predLoss = tf.losses.huber_loss(Y,Pred)

        with tf.name_scope('phi_mean'):
            mean = tf.reduce_mean(Phi)
            phiLoss = tf.square(mean - 0.5) + tf.square(tf.sqrt(tf.reduce_mean(tf.square(Phi - mean))) - 0.08)
            
        with tf.name_scope('loss'):
            alpha = tf.constant(a, dtype=tf.float32) # Scaling factor for magnitude of gradient
            beta  = tf.constant(b, dtype=tf.float32)  # Scaling factor for prediction of next time step 
            gamma  = tf.constant(g, dtype=tf.float32)  # Scaling factor for phi scale invarientp 
            loss = tf.reduce_mean(tf.abs(dotProd / gradMag) + tf.maximum(gradMag - 2, 0) + tf.abs(tf.minimum(gradMag - 1, 0)))
            #loss = tf.reduce_mean(tf.abs(dotProd)) + alpha * gradLoss + gamma * phiLoss
            #loss = tf.reduce_mean(tf.abs(dotProd)) + alpha * gradLoss + beta * predLoss + gamma * phiLoss
            
        with tf.name_scope('train'):
            train_step = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss)

        with tf.name_scope('summaries'):
            Phi_0 = tf.slice(Phi, [0,0], [4223,1])
            Phi_1 = tf.slice(Phi, [4223 * 1 - 1, 0], [4223, 1])
            Phi_2 = tf.slice(Phi, [4223 * 2 - 1, 0], [4223, 1])
            Phi_3 = tf.slice(Phi, [4223 * 3 - 1, 0], [4223, 1])
            Phi_4 = tf.slice(Phi, [4223 * 4 - 1, 0], [4223, 1])
            Phi_5 = tf.slice(Phi, [4223 * 5 - 1, 0], [4223, 1])
            Phi_6 = tf.slice(Phi, [4223 * 6 - 1, 0], [4223, 1])
            Phi_7 = tf.slice(Phi, [4223 * 7 - 1, 0], [4223, 1])
            

    # Create summary statistics outside of GPU scope
    variable_summaries(Phi, "PhiSummary")
    variable_summaries(Phi_0, "Phi_earth")
    variable_summaries(Phi_1, "Phi_jupiter")
    variable_summaries(Phi_2, "Phi_mars")
    variable_summaries(Phi_3, "Phi_mercury")
    variable_summaries(Phi_4, "Phi_neptune")
    variable_summaries(Phi_5, "Phi_saturn")
    variable_summaries(Phi_6, "Phi_uranus")
    variable_summaries(Phi_7, "Phi_venus")
    #variable_summaries(Pred, "Prediction")
    variable_summaries(gradPhi, "GradPhi")
    variable_summaries(dotProd, "DotProduct")
    variable_summaries(gradMag, "GradMagnitude")
    tf.summary.scalar("GradLoss", gradLoss)
    #tf.summary.scalar("PredictiveLoss", predLoss)
    tf.summary.scalar("PhiLoss", phiLoss)
    tf.summary.scalar("Cost", loss)

    # Collect summary stats
    merged = tf.summary.merge_all()

    # Train the model
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        
        timeStr = datetime.datetime.today().strftime('%Y-%m-%d_%H-%M')
        train_writer = tf.summary.FileWriter('./train/alpha-' + str(a) + 'beta-' + str(b) + '/' + timeStr, sess.graph)

        for epoch in range(train_epoch):
            if epoch > pre_train_steps:
                if epoch % summary_step == 0:
                    summary = sess.run([merged])[0]
                    train_writer.add_summary(summary, epoch)

                if epoch % display_step == 0:
                    loss_ = sess.run(loss)
                    print(loss_)

            sess.run([train_step])


main()