import os
import math
import random

import tensorflow as tf
import numpy as np
from functools import reduce
from plotlyGraph import fit_linear

from dataLoader.planets import get_data_, get_data_segmented

J = os.path.join
E = os.path.exists


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
        summaries.append(tf.summary.histogram('histogram', var))
    return summaries


def singleLayer(X, outDim=50, name=False):
    # Hidden layers
    if name:
        head = tf.layers.dense(X, outDim, activation=tf.nn.sigmoid, use_bias=True, name="dense_1")
    else:
        head = tf.layers.dense(X, outDim, activation=tf.nn.sigmoid, use_bias=True)
    return head


def trippleLayer(X, outDim=16, name=False):
    head = tf.layers.dense(X, 64, activation=tf.nn.sigmoid, name="dense_1", use_bias=True)
    head = tf.layers.dense(head, 2, activation=tf.nn.sigmoid, name="dense_2", use_bias=True)
    head = tf.layers.dense(head, outDim, activation=tf.nn.sigmoid, name="dense_3", use_bias=True)
    return head


#######################
#             ...xxx...
train_epoch =  16000000
display_step =    10000
summary_step =    10000
checkpoint_int = 100000
pre_train_steps  = 1000
resample_step =   50000
#             ...xxx...
viz_step = 999999999999999
#######################
a = 0.001  # GradNorm Weight
b = 0.000001  # Prediction Weight
g = 0.001  # Scale for Phi
lr = 0.01  # Learning Rate
#######################

a = 0.0001 # GradNorm Weight
b = 0.00000 # Prediction Weight
g = 0.00505  # Scale for Phi
lr = 0.004  # Learning Rate


def make_save_prefix(saveDir, saveType, a, b, g, lr, base_net):
    return J('.', saveDir, 'alpha-' + str(a) + 'beta-' + str(b) + 'gama-' + str(g) + 'lr-' + str(lr) + base_net, saveType)


def train_model(a, b, g, lr, train_epoch, saveDir=None):
    #return random.sample(range(777), 5)

    # Load data
    # scale, offset, (train_Xp, _, train_X, _, train_F, _, train_Y, _), benchmark = get_data_segmented(shuffle=False, seed=42)
    scale, offset, (train_Xp, _, train_X, _, train_F, _, train_Y, _), benchmark = get_data_(shuffle=False)
    _, _, (test_Xp, _, test_X, _, test_F, _, test_Y, _), _ = get_data_(shuffle=False)

    # Load data onto GPU memory - ensure network layers have GPU support
    # with tf.device('/gpu:0'):
    if True:
        tf.reset_default_graph()
        # # Define GPU constants
        # X = tf.identity(tf.constant(train_X, dtype= tf.float32))
        # F = tf.identity(tf.constant(train_F, dtype= tf.float32))
        # if (b > 0):
        #     Y = tf.identity(tf.constant(train_Y, dtype= tf.float32))

        # Zero vector
        zero = tf.Variable(tf.zeros([33776, 4], dtype=tf.float32))

        # Define placeholders
        Xp = tf.placeholder_with_default(shape=[None, 4], name='Xp', input=zero)
        X = tf.placeholder_with_default(shape=[None, 4], name='X', input=zero)
        F = tf.placeholder_with_default(shape=[None, 4], name='F', input=zero)
        if b > 0:
            Y = tf.placeholder_with_default(shape=[None, 4], name='Y', input=zero)

        ## Define network
        with tf.name_scope('Base_Network'):
            with tf.variable_scope("auto_weight_sharing") as varScope:
                base_net_type = 'single_layer'
                baseNetwork = singleLayer(X, outDim=32, name=True)
                # baseNetwork = trippleLayer(X, outDim=32, name=True)
                varScope.reuse_variables()
                baseNetPrev = singleLayer(Xp, outDim=32, name=True)
                # baseNetPrev = trippleLayer(Xp, outDim=32, name=True)

        with tf.name_scope('Phi'):
            Phi = singleLayer(baseNetwork, outDim=1)

        if (b > 0):
            predNetwork = tf.concat([
                baseNetwork,
                baseNetPrev], axis=-1)
            with tf.name_scope('Prediction'):
                Pred = singleLayer(predNetwork, outDim=4)

        ## Define Loss
        with tf.name_scope('gradPhi'):
            gradPhi = tf.gradients(Phi, [X])[0]

        # Calculate dot product 
        with tf.name_scope('dotProd'):
            # dotProd = tf.reduce_sum(tf.multiply(F, gradPhi), axis=1)
            dotProd = tf.reduce_sum(
                tf.multiply(F, gradPhi) / tf.expand_dims(tf.norm(F, axis=1) * tf.norm(gradPhi, axis=1), dim=1), axis=1)

            # Calculate gradient regularization term
        with tf.name_scope('gradMag'):
            gradMag = tf.norm(gradPhi, axis=1)

        with tf.name_scope('grad_loss'):
            # gradLoss = tf.reduce_mean(tf.square(gradMag - 1))
            # gradLoss = tf.reduce_mean(tf.maximum(gradMag - 2, 0) + tf.maximum(1 - gradMag, 0))
            gradLoss = tf.reduce_mean(tf.maximum(1 - gradMag, 0))

        if (b > 0):
            with tf.name_scope('pred_loss'):
                predLoss = tf.losses.huber_loss(Y, Pred)

        with tf.name_scope('phi_mean'):
            mean = tf.reduce_mean(Phi)
            # phiLoss = tf.square(mean - 0.5) + tf.square(tf.sqrt(tf.reduce_mean(tf.square(Phi - mean))) - 0.08)
            phiLoss = tf.square(mean - 0.5)

        with tf.name_scope('loss'):
            alpha = tf.constant(a, dtype=tf.float32)  # Scaling factor for magnitude of gradient
            beta = tf.constant(b, dtype=tf.float32)  # Scaling factor for prediction of next time step
            gamma = tf.constant(g, dtype=tf.float32)  # Scaling factor for phi scale invarient
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
            planet_values = [tf.slice(Phi, [4222 * i, 0], [4222, 1]) for i in range(8)]
            means = [tf.reduce_mean(planetPhi) for planetPhi in planet_values]
            stratified_var = reduce(lambda x, y: (x + y) / 2,
                                    [tf.sqrt(tf.reduce_mean(tf.square(var - mean))) for (var, mean) in
                                     zip(planet_values, means)])

    # Create summary statistics outside of GPU scope
    variable_summaries(Phi, "PhiSummary")
    tf.summary.scalar("PhiByPlanetVar", stratified_var, )
    variable_summaries(gradPhi, "GradPhi")
    variable_summaries(dotProd, "DotProduct")
    variable_summaries(gradMag, "GradMagnitude")
    tf.summary.scalar("GradLoss", gradLoss)
    if (b > 0):
        tf.summary.scalar("PredictiveLoss", predLoss)
        variable_summaries(Pred, "Prediction")
    tf.summary.scalar("PhiLoss", phiLoss)
    loss_summary = tf.summary.scalar("Cost", loss)

    # Collect summary stats for train variables
    merged = tf.summary.merge_all()

    # Create list of merged summaries to visualize together on a single graph
    summary_lables = ['earth', 'jupiter', 'mars', 'mercury', 'neptune', 'saturn', 'uranus', 'venus']
    summary_lables = [saveDir + '/train/' + planet for planet in summary_lables]
    with tf.name_scope('planets'):
        summary_list = [tf.summary.merge(variable_summaries_list(planet_val, summary_label)) for
                        (planet_val, summary_label) in zip(planet_values, summary_lables)]

    # Create checkpoint saver
    saver = tf.train.Saver()

    # Train the model
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        # timeStr = datetime.datetime.today().strftime('%Y-%m-%d_%H-%M')
        train_writer = tf.summary.FileWriter(make_save_prefix(saveDir, 'train', a, b, g, lr, base_net_type), sess.graph)
        test_writer = tf.summary.FileWriter(make_save_prefix(saveDir, 'test', a, b, g, lr, base_net_type), sess.graph)
        writers = [tf.summary.FileWriter(name) for name in summary_lables]
        dic = {'Xp:0': train_Xp, 'X:0': train_X, 'F:0': train_F}
        test_dic = {'Xp:0': test_Xp, 'X:0': test_X, 'F:0': test_F}
        if b > 0:
            dic['Y:0'] = train_Y
            test_dic['Y:0'] = test_Y


        mses = []
        ratios = []

        for epoch in range(train_epoch + 1):

            if epoch > pre_train_steps:

                # Re-sample training data during training
                if epoch % resample_step == 0:
                    _, _, (train_Xp, _, train_X, _, train_F, _, train_Y, _), _ = get_data_(shuffle=False)
                    dic = {'Xp:0': train_Xp, 'X:0': train_X, 'F:0': train_F}

                # Compute accuracy and fit linear model
                if epoch % summary_step == 0:
                    summary = sess.run([merged], feed_dict=test_dic)[0]
                    test_writer.add_summary(summary, epoch)
                    loss_ = sess.run([loss_summary], feed_dict=dic)[0]
                    train_writer.add_summary(loss_, epoch)
                    [writer.add_summary(sess.run([summary], feed_dict=test_dic)[0], epoch) for (writer, summary) in
                     zip(writers, summary_list)]
                    phi_ = sess.run([Phi], feed_dict=test_dic)[0]

                    try:
                        mse, ratio = fit_linear(offset, scale, test_X[:, 0], test_X[:, 1], test_X[:, 2], test_X[:, 3],
                                         phi_.ravel())
                        mse_summary = tf.Summary()
                        mse_ratio_summary = tf.Summary()
                        mse_summary.value.add(tag='mse', simple_value=mse)
                        mses.append(mse)
                        test_writer.add_summary(mse_summary, epoch)
                        mse_ratio_summary.value.add(tag='ratio', simple_value=ratio)
                        ratios.append(ratio)
                        test_writer.add_summary(mse_ratio_summary, epoch)
                    except RuntimeError:
                        pass

                # Compute loss
                if epoch % display_step == 0:
                    loss_ = sess.run(loss, feed_dict=dic)
                    print(loss_, epoch)

                if epoch % checkpoint_int == 0 and saveDir is not None:
                    saver.save(sess, save_path=make_save_prefix(saveDir, 'network', a, b, g, lr, base_net_type)+str(epoch))

                # if epoch % viz_step == 0:
                #     phi = sess.run([Phi],feed_dict=viz_dic)
                #     np.save('./train/'+str(epoch), phi)
                #     phi = sess.run([Phi],feed_dict=dic)
                #     np.save('./train/p'+str(epoch),np.array([train_X[:,0], train_X[:,1], phi]))

            sess.run([train_step], feed_dict=dic)

            # sess.run([train2_step])
    return mses, ratios


if __name__ == "__main__":
    saveDir = input("Name this run...")
    train_model(a, b, g, lr, train_epoch, saveDir)
