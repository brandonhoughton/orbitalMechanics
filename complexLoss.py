import os
import math
import random
import tensorflow as tf
import numpy as np
import datetime
from functools import reduce

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

    
def variable_summaries_list(var, groupName):
  """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
  with tf.name_scope(groupName):
    summaries = []
    mean = tf.reduce_mean(var)
    summaries.append(tf.summary.scalar('mean', mean))
    with tf.name_scope('stddev'):
      stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
    summaries.append(tf.summary.scalar('stddev', stddev))
    #summaries.append(tf.summary.scalar('max', tf.reduce_max(var)))
    #summaries.append(tf.summary.scalar('min', tf.reduce_min(var)))
    summaries.append(tf.summary.histogram('histogram', var))
    return summaries

def singleLayer(X, outDim = 50):
    #Hidden layers
    head = tf.layers.dense(X, outDim, activation=tf.nn.sigmoid, use_bias=True)
    return head

def trippleLayer(X, outDim = 16):
    head = tf.layers.dense(X, 64, activation=tf.nn.sigmoid, use_bias=True)
    head = tf.layers.dense(head, 24, activation=tf.nn.sigmoid, use_bias=True)
    head = tf.layers.dense(head, outDim, activation=tf.nn.sigmoid, use_bias=True)
    return head


#######################
train_epoch =    500000
display_step =     1000
summary_step =     2000
checkpoint_int = 100000
pre_train_steps  = 1000
viz_step         = 5000
#######################
a = 0.01 # GradNorm Weight
b = 0.00 # Prediction Weight
g = 0.01 # Scale for Phi
lr = 0.01 # Learning Rate
#######################
    
def main():

    # Load data
    scale, offset, (train_X, test_X, train_F, test_F, train_Y, test_Y), benchmark = get_data(shuffle=False)

    # Load data onto GPU memory - ensure network layers have GPU support
    #with tf.device('/gpu:0'):
    if True:
        # # Define GPU constants
        # X = tf.identity(tf.constant(train_X, dtype= tf.float32))
        # F = tf.identity(tf.constant(train_F, dtype= tf.float32))
        # if (b > 0):
        #     Y = tf.identity(tf.constant(train_Y, dtype= tf.float32))

        # Define placeholders
        X = tf.placeholder(dtype= tf.float32, shape=[None, 4])
        F = tf.placeholder(dtype= tf.float32, shape=[None, 4])
        if (b > 0):
            Y = tf.placeholder(dtype= tf.float32, shape=[None, 4])

        ## Define network
        with tf.name_scope('Base_Network'):
            baseNetwork = singleLayer(X, outDim=32)

        with tf.name_scope('Phi'):
            Phi = singleLayer(baseNetwork, outDim = 1)

        if(b > 0):    
            with tf.name_scope('Prediction'):
                Pred = singleLayer(baseNetwork, outDim = 4)

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
            #gradLoss = tf.reduce_mean(tf.square(gradMag - 1))       
            gradLoss = tf.reduce_mean(tf.maximum(gradMag - 2, 0) + tf.maximum(1 - gradMag, 0))

        if (b > 0):
            with tf.name_scope('pred_loss'):
                predLoss = tf.losses.huber_loss(Y,Pred)

        with tf.name_scope('phi_mean'):
            mean = tf.reduce_mean(Phi)
            #phiLoss = tf.square(mean - 0.5) + tf.square(tf.sqrt(tf.reduce_mean(tf.square(Phi - mean))) - 0.08)
            phiLoss = tf.square(mean - 0.5)
            
        with tf.name_scope('loss'):
            alpha = tf.constant(a, dtype=tf.float32) # Scaling factor for magnitude of gradient
            beta  = tf.constant(b, dtype=tf.float32)  # Scaling factor for prediction of next time step 
            gamma  = tf.constant(g, dtype=tf.float32)  # Scaling factor for phi scale invarientp 
            loss = tf.reduce_mean(tf.abs(dotProd))

            if (a > 0):
                loss += alpha * gradLoss
            if (b > 0):
                loss += beta * predLoss
            if (g > 0):
                loss += gamma * phiLoss
            
        with tf.name_scope('train'):
            train_step = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss)

        with tf.name_scope('summaries'):
            planet_values = [tf.slice(Phi, [4223 * i, 0], [4223, 1]) for i in range(8)]
            means = [tf.reduce_mean(planetPhi) for planetPhi in planet_values]
            stratified_var = reduce(lambda x,y: (x + y) / 2, [tf.sqrt(tf.reduce_mean(tf.square(var - mean))) for (var, mean) in zip(planet_values, means)])

            

    # Create summary statistics outside of GPU scope
    variable_summaries(Phi, "PhiSummary")
    tf.summary.scalar("PhiByPlanetVar", stratified_var,)

    #variable_summaries(Pred, "Prediction")
    variable_summaries(gradPhi, "GradPhi")
    variable_summaries(dotProd, "DotProduct")
    variable_summaries(gradMag, "GradMagnitude")
    tf.summary.scalar("GradLoss", gradLoss)
    #tf.summary.scalar("PredictiveLoss", predLoss)
    tf.summary.scalar("PhiLoss", phiLoss)
    tf.summary.scalar("Cost", loss)

    # Collect summary stats for train variables
    merged = tf.summary.merge_all()

    #Create list of merged summaries to visualize together on a single graph
    summary_lables = ['earth', 'jupiter', 'mars', 'mercury', 'neptune', 'saturn', 'uranus', 'venus'] 
    summary_lables = ['./train/' + planet for planet in summary_lables]
    with tf.name_scope('planets'):
        summary_list = [tf.summary.merge(variable_summaries_list(planet_val, summary_label)) for (planet_val, summary_label) in zip(planet_values, summary_lables)]


    # Create checkpoint saver
    saver = tf.train.Saver()

    # Visualization mesh:
    nx, ny = (100, 100)

    x = np.linspace(-1, 1, nx)
    y = np.linspace(-1, 1, ny)
    X, Y = np.meshgrid(x, y)
    X = np.reshape(X, (-1))
    Y = np.reshape(Y, (-1))
    T = np.arctan2(Y, X) + (math.pi / 2.0)
    R = 1.5 + np.sqrt((Y) ** 2 + (X) ** 2)
    U, V = R * np.cos(T), R * np.sin(T)
    viz_dic = {'X':np.array([X, Y, U, V])}

    # Train the model
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        
        #timeStr = datetime.datetime.today().strftime('%Y-%m-%d_%H-%M')
        train_writer = tf.summary.FileWriter('./train/alpha-' + str(a) + 'beta-' + str(b) + 'gama-' + str(g), sess.graph)
        writers = [tf.summary.FileWriter(name) for name in summary_lables]
        dic = {'X':train_X, 'F':train_F}

        for epoch in range(train_epoch):
            
            if epoch > pre_train_steps:
                if epoch % summary_step == 0:
                    summary = sess.run([merged],feed_dict=dic)[0]
                    train_writer.add_summary(summary, epoch)
                    [writer.add_summary(sess.run([summary],feed_dict=dic)[0], epoch) for (writer,summary) in zip(writers,summary_list)]

                if epoch % display_step == 0:
                    loss_ = sess.run(loss,feed_dict=dic)
                    print(loss_, epoch)

                if epoch % checkpoint_int == 0:
                    saver.save(sess,save_path='./network/'+ str(epoch))

                if epoch % viz_step == 0:
                    phi = sess.run([Phi],feed_dict=viz_dic)
                    np.save('./np/'+str(epoch), phi)
                    

            sess.run([train_step],feed_dict=dic)



main()