import json
import os
import tensorflow as tf
import numpy as np
import traceback
from tensorflow import keras
from tensorboard.plugins.beholder import Beholder
from src.dataLoader.turbulence import Turbulence, LARGE_DATASET, TEST_DATASET_5, datasets

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
        head = tf.nn.relu(head)

    # Skip layer
    return head


def ped(X, pred_len, skip_depth=3, num_features=16, kernel_size=5):
    # Input 50 x 50 X history_len patch
    # Map this patch to match the residual cell size
    head = tf.layers.conv2d(
        X, filters=num_features, kernel_size=1, activation=tf.nn.relu, name='map_input', padding='same')

    head = tf.scan(
        fn=lambda acc, _: residual_cell(acc, skip_depth, num_features, kernel_size),
        elems=tf.zeros(pred_len), initializer=head)

    head = tf.map_fn(
        fn=lambda elem:
            tf.layers.conv2d(inputs=elem, filters=1, kernel_size=1, activation=tf.nn.relu, name='map_output'),
        elems=head
    )

    head = tf.squeeze(head)

    return head


#######################
train_batch = 200000
summary_step = 500
validation_step = 5000
checkpoint_int = 20000
pre_train_steps = 500
save_pred_steps = 10000
#######################
use_split_pred = False
a = 0.0001  # GradNorm Weight
b = 0.00000000  # Prediction Weight
g = 0.005  # Scale for Phi
lr = 0.0005  # Learning Rate
#######################

########################################################################################################################

_net_name = 'PDE_3-skip_1-cell_resnet'
_save_dir = os.path.join('experiments', 'turbulence', 'recurrent_scaled_mse')


########################################################################################################################

def train(net_name=_net_name, save_dir=_save_dir, dataset_idx=LARGE_DATASET, loader=None, num_batches=train_batch,
          pixel_dropout=None):

    history_length = 20
    pred_length = 40

    LOG_DIR = J('.', save_dir, net_name + '_' + datasets[dataset_idx] +
                '_{}-lr_{}-hist_{}-pred'.format(lr, history_length, pred_length))

    # Start beholder
    beholder = Beholder(LOG_DIR)

    # Load data

    if loader is None:
        loader = Turbulence(history_length=history_length, pred_length=pred_length, dataset_idx=dataset_idx)
    else:
        pred_length = loader.pred_length

    # Load data onto GPU memory - ensure network layers have GPU support
    config = tf.ConfigProto(log_device_placement=True)
    config.gpu_options.allow_growth = True
    # config.allow_soft_placement = True
    with tf.Session(config=config) as sess:

        # sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

        with tf.device('/cpu:0'):
            test = tf.placeholder_with_default(tf.constant(False, dtype=tf.bool), name="testing_flag", shape=())
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

                # # Get any input noise if present
                # if loader.get_input_noise() is not None:
                #     noise = tf.constant(dtype=tf.float32, value=loader.get_input_noise())
                #     input_data = data + noise
                # else:
                #     input_data = data
                #
                # # Handle dropout if present
                # if pixel_dropout is not None:
                #     input_data = tf.layers.dropout(input_data, rate=pixel_dropout)

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
                size = [50, 50, pred_length]
                # Y = tf.map_fn(lambda i: data[0:50, 0:50, history_length:history_length+pred_length], tf.zeros(64), dtype=tf.float32)

                Y = tf.map_fn(lambda region: tf.slice(data, region[1], size), regions, dtype=tf.float32)

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
        Pred = ped(X, pred_length)

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

        with tf.name_scope('train'):
            adam = tf.train.AdamOptimizer(learning_rate=lr)
            grads = adam.compute_gradients(pred_loss)
            train_step = adam.apply_gradients(grads)

        tf.summary.histogram("Loss Histogram", loss_over_time)
        # tf.summary.histogram("Confidence Histogram", conf)
        # tf.summary.histogram("Weighted Histogram", conf_losses_over_time)

        tf.summary.scalar("PredictiveLoss", pred_loss)
        # tf.summary.scalar("AvgConfidence", avg_cong)
        # tf.summary.scalar("ConfWeightedLoss", conf_weighted_loss)

        # for grad in grads:
        #     tf.summary.scalar("NormGradL1" + str(grad), tf.reduce_mean(tf.abs(grad)))
        #     tf.summary.histogram("grad" + str(grad), grad)

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
            for batch in range(num_batches + 1):
                if batch > pre_train_steps and batch % summary_step == 0:
                    loss, summary, accuracy = sess.run([pred_loss, merged, loss_over_time])
                    # loss, confidence, summary = sess.run([pred_loss, avg_cong, merged])
                    train_writer.add_summary(summary, batch)
                    train_accuracy[str(batch)] = accuracy
                    print(loss, batch)
                elif batch % summary_step == 0:
                    loss, summary = sess.run([pred_loss, merged])
                    # loss, confidence, summary = sess.run([pred_loss, avg_cong, merged])
                    print('(', loss, batch, ')')
                else:
                    sess.run(train_step)
                    beholder.update(sess)

                if batch > pre_train_steps and batch % save_pred_steps == 0:
                    flags = dict({'testing_flag:0': True})
                    summaries, prediction, label, input_with_noise, accuracy = sess.run(
                        [merged_with_imgs, Pred, Y, X, pred_loss], feed_dict=flags)
                    for summary in summaries:
                        test_writer.add_summary(summary, batch)
                    input_sequences[str(batch)] = np.array(input_with_noise)
                    predicted_sequences[str(batch)] = np.array(prediction)
                    label_sequences[str(batch)] = np.array(label)

                if batch > pre_train_steps and batch % validation_step == 0:
                    flags = dict({'testing_flag:0': True})
                    summaries, accuracy = sess.run([merged_with_imgs, loss_over_time], feed_dict=flags)
                    for summary in summaries:
                        test_writer.add_summary(summary, batch)
                    validation_accuracy[str(batch)] = np.array(accuracy)

                if batch % checkpoint_int == 0:
                    saver.save(sess, save_path=J(LOG_DIR, 'network', str(batch)))
        finally:
            np.savez_compressed(J(LOG_DIR, 'train_accuracy_by_time'), **train_accuracy)
            np.savez_compressed(J(LOG_DIR, 'validation_accuracy_by_time'), **validation_accuracy)
            np.savez_compressed(J(LOG_DIR, 'predictions'), **predicted_sequences)
            np.savez_compressed(J(LOG_DIR, 'labels'), **label_sequences)
            np.savez_compressed(J(LOG_DIR, 'inputs'), **input_sequences)


if __name__ == "__main__":
    os.chdir("..")
    train()
