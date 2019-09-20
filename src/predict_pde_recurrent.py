import functools
import json
import os
import tensorflow as tf
import numpy as np
import traceback
import tqdm
from tensorflow import keras
from tensorboard.plugins.beholder import Beholder
from src.dataLoader.turbulence import Turbulence, LARGE_DATASET, TEST_DATASET_5, datasets

from tensorflow.python.util import deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False

J = os.path.join
E = os.path.exists


#   Architectures   #


def residual_cell(activation, skip_depth, num_features, kernel_width):
    num_filters = num_features
    kernel_size = (kernel_width, kernel_width)

    # Pass activation to last layer and cell layers
    head = activation

    # Cell layers
    for i in range(skip_depth):
        head = tf.layers.conv2d(
            head, num_filters, kernel_size, name='res_layer_{}'.format(i),
            padding='same')
        if i == skip_depth - 1:
            head += activation
        head = tf.nn.leaky_relu(head)

    # Skip layer
    return head


def pde(X, pred_len, skip_depth=3, num_features=16, kernel_size=5, encoder_kernel_size=1):
    # Input 50 x 50 X history_len patch
    # Map this patch to match the residual cell size
    head = tf.layers.conv2d(
        X, filters=num_features, kernel_size=encoder_kernel_size, activation=tf.nn.leaky_relu, name='map_input',
        padding='valid')

    head = tf.scan(
        fn=lambda acc, _: residual_cell(acc, skip_depth, num_features, kernel_size),
        elems=tf.zeros(pred_len), initializer=head, swap_memory=True)

    head = tf.map_fn(
        fn=lambda elem:
            tf.layers.conv2d(inputs=elem, filters=1, kernel_size=1, activation=tf.nn.leaky_relu, name='map_output'),
        elems=head
    )

    head = tf.squeeze(head)

    return head


def runge_kutta_pde(X, pred_len, skip_depth=3, num_features=16, kernel_size=5, encoder_kernel_size=1):
    # Input 50 x 50 X history_len patch
    # Map this patch to match the residual cell size
    head = tf.layers.conv2d(
        X, filters=num_features, kernel_size=encoder_kernel_size, activation=tf.nn.relu, name='map_input',
        padding='valid')

    def rk4(y):
        k1 = residual_cell(y, skip_depth, num_features, kernel_size)
        k2 = residual_cell(y + 0.5 * k1, skip_depth, num_features, kernel_size)
        k3 = residual_cell(y + 0.5 * k2, skip_depth, num_features, kernel_size)
        k4 = residual_cell(y + k3, skip_depth, num_features, kernel_size)
        return (k1 + k2 + k3 + k4) / 4

    head = tf.scan(
        fn=lambda acc, _: rk4(acc),
        elems=tf.zeros(pred_len), initializer=head, swap_memory=True)

    head = tf.map_fn(
        fn=lambda elem:
            tf.layers.conv2d(inputs=elem, filters=1, kernel_size=1, activation=tf.nn.relu, name='map_output'),
        elems=head
    )

    head = tf.squeeze(head)

    return head


#######################
train_batch = 50000
summary_step = 500
validation_step = 2000
checkpoint_int = 10000
pre_train_steps = 500
save_pred_steps = 10000
#######################
use_split_pred = False
a = 0.0001  # GradNorm Weight
b = 0.00000000  # Prediction Weight
g = 0.005  # Scale for Phi
lr = 0.001  # Learning Rate
#######################

########################################################################################################################

_net_name = 'lin_relu_cell'
_save_dir = os.path.join('experiments', 'turbulence', 'pde')


########################################################################################################################

def train(net_name=_net_name,
          save_dir=_save_dir,
          dataset_idx=LARGE_DATASET,
          loader=None,
          num_batches=train_batch,
          pixel_dropout=None,
          conv_width=5,
          history_length=5,
          pred_length=40,
          encoder_kernel_size=1,
          network=None,
          retrain=False,
          multi_pred_len=False,
          starting_batch=0):

    # Ensure that we don't carry over any variables from previous sessions
    tf.reset_default_graph()

    if multi_pred_len:
        LOG_DIR = J('.', save_dir, net_name +
                '_{}-lr_{}-hist'.format(lr, history_length))
    else:
        LOG_DIR = J('.', save_dir, net_name +
                    '_{}-lr_{}-hist_{}-pred'.format(lr, history_length, pred_length))

    # Start beholder
    # beholder = Beholder(LOG_DIR)

    # Load data
    if loader is None:
        loader = Turbulence(history_length=history_length, pred_length=pred_length, dataset_idx=dataset_idx)
    else:
        pred_length = loader.pred_length

    # Load data onto GPU memory - ensure network layers have GPU support
    config = tf.ConfigProto()
    # config.log_device_placement=True
    # config.gpu_options.allow_growth = True
    # config.allow_soft_placement = True
    with tf.Session(config=config) as sess:

        # sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

        with tf.device('/cpu:0'):
            test = tf.placeholder_with_default(tf.constant(False, dtype=tf.bool), name="testing_flag", shape=())
            is_training = tf.logical_not(test)
            regions = tf.cond(test, strict=True,
                              true_fn=lambda: loader.get_test_region_batch,
                              false_fn=lambda: loader.get_train_region_batch)

        # if True:
        with tf.device('/gpu:0'):
            with tf.name_scope('Data'):
                # Data does not fit into tensorflow data pipeline - so we split it later using a tensorflow slice op
                data_value = loader.get_data()
                data = tf.constant(dtype=tf.float32, value=data_value)
                input_data = data

                # Get any input noise if present
                if loader.get_input_noise() is not None:
                    noise = tf.constant(dtype=tf.float32, value=loader.get_input_noise())
                    input_data = data + noise
                else:
                    input_data = data

                # Handle dropout if present
                if pixel_dropout is not None:
                    input_data = tf.layers.dropout(input_data, rate=pixel_dropout, training=True)

            ############################################################################################################

            with tf.name_scope('Input'):
                # Simple 50 x 50 region with 20 frames of history.

                # X = tf.gather_nd()
                # rrr = regions
                # print(rrr)
                # region = tf.map_fn(lambda region: region[0], regions, dtype=tf.float32)
                # print(region)
                # X = tf.map_fn(lambda i: input_data[0:50, 0:50, 0:history_length], tf.zeros(64), dtype=tf.float32)
                # print(X)
                X = tf.map_fn(lambda region: tf.slice(input_data, region[0], [50, 50, history_length]), regions, dtype=tf.float32)

                # Tensorflow wants time first, before transpose we have [batch, u, v, time]:[0, 1, 2, 3]
                # X = tf.transpose(X, perm=[3, 0, 1, 2])

                # Ensure that the shape is well defined ( mapping slice operations is evaluated as a dynamic size)
                # X = tf.reshape(X, (20, -1, 50 * 50))  # Flattened for fully connected layers

            with tf.name_scope('Label'):
                over = encoder_kernel_size - 1
                size = [50 - over, 50 - over, pred_length]
                Y = tf.map_fn(lambda region: tf.slice(data, [region[1][0] + over//2, region[1][1] + over//2,
                                                             region[1][2]], size), regions, dtype=tf.float32)

                # size = [50, 50, pred_length]
                # # Y = tf.map_fn(lambda i: data[0:50, 0:50, history_length:history_length+pred_length], tf.zeros(64), dtype=tf.float32)
                #
                # Y = tf.map_fn(lambda region: tf.slice(data, region[1], size), regions, dtype=tf.float32)

                # Map the label to the same time first ordering
                Y = tf.transpose(Y, perm=[3, 0, 1, 2])

                # outDim = [pred_length, -1, 50 * 50]
                # Y = tf.reshape(Y, outDim)

        print("Input shape:", X.shape)
        print("Output shape:", Y.shape)
        # print_op = tf.Print(X,[X])

        ################################################################################################################

        #########################
        #     Define network    #
        #########################
        if network is None:
            Pred = pde(X, pred_length, kernel_size=conv_width, encoder_kernel_size=encoder_kernel_size)
        else:
            Pred = network(X, pred_length, is_training)

        print("Network shape:", Pred.shape)

        # with tf.name_scope('Prediction'):
        #
        #     # Map the enlarger network over the first axis (time)
        #     Pred = tf.map_fn(lambda x: tf.layers.dense(x, num_out, activation=None), state)
        #
        #     # [200, 64, 50 * 50]              0,  1,  2,  3
        #     Pred = tf.reshape(Pred, [pred_length, -1, 50, 50])  # Reshape the output to be width x height x 1
        #     # [200, 64, 50, 50]
        #     # Pred = tf.transpose(Pred, perm=[1, 2, 3, 0])

        ################################################################################################################

        #########################
        #      Define loss      #
        #########################

        with tf.name_scope('loss'):
            # losses = tf.losses.huber_loss(Y, Pred, reduction=tf.losses.Reduction.NONE)
            losses = tf.losses.mean_squared_error(Y, Pred, reduction=tf.losses.Reduction.NONE)
            losses = tf.reduce_mean(losses, axis=[1, 2, 3])  # Reduce image dims
            loss_over_time = losses
            # loss_over_time = tf.reduce_mean(losses, axis=[1])  # Don't reduce time major axis

            # conf_losses = tf.map_fn(lambda s: (1-s[0])*1 + abs(s[0])*(1000000 * s[1]**2), tf.stack([conf, losses]))
            # conf_losses = tf.map_fn(lambda s: (1-s[0])*(s[1] + 0.009) + (s[0])*(10*s[1]), tf.stack([conf, losses]))

            # conf_losses = tf.map_fn(lambda s: (0.009 / (s[0]**2+1)) + (0.01 + 10 * s[0]**2)*s[1], tf.stack([conf, losses], axis=-1))
            # conf_losses_over_time = conf_losses  # tf.reduce_mean(conf_losses, axis=[1])  # Don't reduce time major axis

            pred_loss = tf.reduce_mean(loss_over_time)
            # conf_weighted_loss = tf.reduce_mean(conf_losses_over_time)

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            adam = tf.train.AdamOptimizer(learning_rate=lr)
            grads = adam.compute_gradients(pred_loss)
            train_step = adam.apply_gradients(grads)

            # # Add summaries for gradients
            # for grad in grads:
            #     tf.summary.scalar("NormGradL1" + str(grad), tf.reduce_mean(tf.abs(grad)))
            #     tf.summary.histogram("grad" + str(grad), grad)

        tf.summary.histogram("Loss Histogram", loss_over_time)
        # tf.summary.histogram("Confidence Histogram", conf)
        # tf.summary.histogram("Weighted Histogram", conf_losses_over_time)

        tf.summary.scalar("PredictiveLoss", pred_loss)
        # tf.summary.scalar("AvgConfidence", avg_cong)
        # tf.summary.scalar("ConfWeightedLoss", conf_weighted_loss)

        for weight in tf.trainable_variables():
            tf.summary.scalar("MeanWeight" + str(weight), tf.reduce_mean(weight))
            tf.summary.histogram("w" +str(weight), weight)

        # Collect summary stats for train variables
        merged = tf.summary.merge_all()

        merged_with_imgs = \
            [merged,
             tf.summary.image('Predicted_t0', tf.expand_dims(Pred[0, :, :, :], axis=-1), max_outputs=5),
             tf.summary.image('Label', tf.expand_dims(Y[0, :, :, :], axis=-1), max_outputs=5),
             tf.summary.image('Error',
                              tf.expand_dims(Pred[0, :, :, :], axis=-1) - tf.expand_dims(Y[0, :, :, :], axis=-1),
                              max_outputs=5),
             tf.summary.image('Mean Abs Error', tf.expand_dims(
                 tf.reduce_mean(abs(tf.expand_dims(Pred[0, :, :, :], axis=-1) - tf.expand_dims(Y[0, :, :, :], axis=-1)),
                                axis=0), axis=0), max_outputs=1),
             # ##
             ]
        #     tf.summary.image('Predicted_t59', tf.expand_dims(Pred[59, :, :, :], axis=-1), max_outputs=5),
        #     tf.summary.image('Label_t59', tf.expand_dims(Y[59, :, :, :], axis=-1), max_outputs=5),
        #     tf.summary.image('Error_t59', tf.expand_dims(Pred[59, :, :, :], axis=-1) - tf.expand_dims(Y[59, :, :, :], axis=-1), max_outputs=5),
        #     tf.summary.image('Mean Abs Error_t59', tf.expand_dims(tf.reduce_mean(abs(tf.expand_dims(Pred[59, :, :, :], axis=-1) - tf.expand_dims(Y[59, :, :, :], axis=-1)), axis=0), axis=0), max_outputs=1),
        #      ##
        #     tf.summary.image('Predicted_t109', tf.expand_dims(Pred[109, :, :, :], axis=-1), max_outputs=5),
        #     tf.summary.image('Label_t109', tf.expand_dims(Y[109, :, :, :], axis=-1), max_outputs=5),
        #     tf.summary.image('Error_t109', tf.expand_dims(Pred[109, :, :, :], axis=-1) - tf.expand_dims(Y[109, :, :, :], axis=-1), max_outputs=5),
        #     tf.summary.image('Mean Abs Error_t109', tf.expand_dims(tf.reduce_mean(abs(tf.expand_dims(Pred[100, :, :, :], axis=-1) - tf.expand_dims(Y[10, :, :, :], axis=-1)), axis=0), axis=0), max_outputs=1),
        #      ##
        #     tf.summary.image('Predicted_t159', tf.expand_dims(Pred[159, :, :, :], axis=-1), max_outputs=5),
        #     tf.summary.image('Label_t159', tf.expand_dims(Y[159, :, :, :], axis=-1), max_outputs=5),
        #     tf.summary.image('Error_t159', tf.expand_dims(Pred[159, :, :, :], axis=-1) - tf.expand_dims(Y[159, :, :, :], axis=-1), max_outputs=5),
        #     tf.summary.image('Mean Abs Error_t159', tf.expand_dims(tf.reduce_mean(abs(tf.expand_dims(Pred[159, :, :, :], axis=-1) - tf.expand_dims(Y[159, :, :, :], axis=-1)), axis=0), axis=0), max_outputs=1),
        #     ##
        #     tf.summary.image('Predicted_t199', tf.expand_dims(Pred[199, :, :, :], axis=-1), max_outputs=5),
        #     tf.summary.image('Label_t199', tf.expand_dims(Y[199, :, :, :], axis=-1), max_outputs=5),
        #     tf.summary.image('Error_t199', tf.expand_dims(Pred[199, :, :, :], axis=-1) - tf.expand_dims(Y[199, :, :, :], axis=-1), max_outputs=5),
        #     tf.summary.image('Mean Abs Error_t199', tf.expand_dims(tf.reduce_mean(abs(tf.expand_dims(Pred[199, :, :, :], axis=-1) - tf.expand_dims(Y[199, :, :, :], axis=-1)), axis=0), axis=0), max_outputs=1)]

        # Create checkpoint saver
        saver = tf.train.Saver()

        if retrain:
            print('restoring weights')
            # print('Searching for last checkpoint here:', J(LOG_DIR, 'network'))
            saver.restore(sess, tf.train.latest_checkpoint(J(LOG_DIR, 'network')))
        else:
            sess.run(tf.global_variables_initializer())

        train_accuracy = dict()
        validation_accuracy = dict()
        input_sequences = dict()
        predicted_sequences = dict()
        label_sequences = dict()

        # sess.run(print_op)

        # Setup tensorboard logging directories
        train_writer = tf.summary.FileWriter(
            J(LOG_DIR, 'train'), sess.graph)

        test_writer = tf.summary.FileWriter(
            J(LOG_DIR, 'validation'), sess.graph)

        # Train model
        try:
            for batch in tqdm.tqdm_notebook(range(num_batches), smoothing=0.02):
                if batch > pre_train_steps and batch % summary_step == 0:
                    loss, summary, accuracy = sess.run([pred_loss, merged, loss_over_time])
                    # loss, confidence, summary = sess.run([pred_loss, avg_cong, merged])
                    train_writer.add_summary(summary, batch + starting_batch)
                    train_accuracy[str(batch)] = accuracy
                    # tqdm.tqdm.write(' {} {}'.format(loss, batch))
                elif batch % summary_step == 0:
                    loss, summary = sess.run([pred_loss, merged])
                    # loss, confidence, summary = sess.run([pred_loss, avg_cong, merged])
                    # tqdm.tqdm.write('({} {})'.format(loss, batch))
                else:
                    sess.run(train_step)
                    # beholder.update(sess)

                if batch > pre_train_steps and batch % save_pred_steps == 0\
                        or batch == num_batches:
                    flags = dict({'testing_flag:0': True})
                    summaries, prediction, label, input_with_noise, accuracy = sess.run(
                        [merged_with_imgs, Pred, Y, X, pred_loss], feed_dict=flags)
                    for summary in summaries:
                        test_writer.add_summary(summary, batch + starting_batch)
                    input_sequences[str(batch)] = np.array(input_with_noise)
                    predicted_sequences[str(batch)] = np.array(prediction)
                    label_sequences[str(batch)] = np.array(label)

                if batch > pre_train_steps and batch % validation_step == 0:
                    flags = dict({'testing_flag:0': True})
                    summaries, accuracy = sess.run([merged_with_imgs, loss_over_time], feed_dict=flags)
                    for summary in summaries:
                        test_writer.add_summary(summary, batch + starting_batch)
                    validation_accuracy[str(batch + starting_batch)] = np.array(accuracy)

                if batch > pre_train_steps and batch % checkpoint_int == 0:
                    saver.save(sess, save_path=J(LOG_DIR, 'network', str(batch + starting_batch)))
            # Completed Training
            # print('Saving the graph here:', J(LOG_DIR, 'network', str(num_batches + starting_batch)))
            saver.save(sess, save_path=J(LOG_DIR, 'network', str(num_batches + starting_batch)))

        finally:
            np.savez_compressed(J(LOG_DIR, 'train_accuracy_by_time_' + str(batch + starting_batch)), **train_accuracy)
            np.savez_compressed(J(LOG_DIR, 'validation_accuracy_by_time_' + str(batch + starting_batch)),
                                **validation_accuracy)
            np.savez_compressed(J(LOG_DIR, 'predictions_' + str(batch + starting_batch)), **predicted_sequences)
            np.savez_compressed(J(LOG_DIR, 'labels_' + str(batch + starting_batch)), **label_sequences)
            np.savez_compressed(J(LOG_DIR, 'inputs_' + str(batch + starting_batch)), **input_sequences)


if __name__ == "__main__":
    os.chdir("..")
    train()
