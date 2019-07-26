import json
import os
import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorboard.plugins.beholder import Beholder
from src.dataLoader.turbulence import Turbulence, LARGE_DATASET, TEST_DATASET_5, datasets
J = os.path.join
E = os.path.exists

## Architecures ##


def lstm_encode(X, outDim=[250, 250, 250], batchSize = 64):
    print('Encoder input shape:', X.get_shape().as_list())
    rnn_cell = tf.nn.rnn_cell.MultiRNNCell(
        [tf.nn.rnn_cell.LSTMCell(dim, name="encode_" + str(idx)) for (idx, dim) in enumerate(outDim)]
        # [tf.nn.rnn_cell.BasicRNNCell(dim, name="encode_" + str(idx), activation=None) for (idx, dim) in enumerate(outDim)]
        # [tf.nn.rnn_cell.GRUCell(dim, name="encode_" + str(idx)) for (idx, dim) in enumerate(outDim)]
    )
    initial_state = rnn_cell.zero_state(batchSize, dtype=tf.float32)
    # print('Zero state shape:(', initial_state[0].get_shape().as_list(), ',', initial_state[1].get_shape().as_list(), ')')
    output, state = tf.nn.dynamic_rnn(rnn_cell, X, initial_state=initial_state, dtype=tf.float32, time_major=True, parallel_iterations=64)

    # print('Output state shape:(', state[0].get_shape().as_list(), ',', state[1].get_shape().as_list(), ')')

    # We ignore the output of this layer, just need it to build an embedding
    return state


def lstm_decode(X, outDim=[250, 250, 250], batchSize=64, pred_length=200, inputShape=[20, -1, 50 * 50]):
    padded = tf.ones([pred_length, batchSize, 50*50], dtype=tf.float32)
    print('Padded input shape:', padded.get_shape().as_list())

    state = X
    # state = rnn_cell.zero_state(batchSize, dtype=tf.float32)
    # print('State shape:(', X[0].get_shape().as_list(), ',', X[1].get_shape().as_list(), ')')
    # print('Zero state shape:(', state[0].get_shape().as_list(), ',', state[1].get_shape().as_list(), ')')
    rnn_cell = tf.nn.rnn_cell.MultiRNNCell(
        # [tf.nn.rnn_cell.GRUCell(dim, name="decode_" + str(idx)) for (idx, dim) in enumerate(outDim)]
        [tf.nn.rnn_cell.LSTMCell(dim, name="decode_" + str(idx)) for (idx, dim) in enumerate(outDim)]
        # [tf.nn.rnn_cell.BasicRNNCell(dim, name="decode_" + str(idx), activation=None) for (idx, dim) in enumerate(outDim)]
    )
    output, state = tf.nn.dynamic_rnn(rnn_cell, padded, initial_state=state, dtype=tf.float32, time_major=True, parallel_iterations=64)
    print(output.get_shape().as_list())

    # We ignore the state, just  return the output
    return output


def seq2seq(X, outDim= 512, outLen=10, num_layers=3):
    # Create a `layers_stacked_count` of stacked RNNs (GRU cells here).
    cells = []
    for i in range(num_layers):
        with tf.variable_scope('RNN_{}'.format(i)):
            cells.append(tf.nn.rnn_cell.GRUCell(outDim))
            # cells.append(tf.nn.rnn_cell.BasicLSTMCell(...))
    cell = tf.nn.rnn_cell.MultiRNNCell(cells)
    decoder_inputs = [tf.zeros_like(X[0], dtype=tf.float32, name="GO")] + X[:-1]
    dec_outputs, dec_memory = tf.contrib.legacy_seq2seq.basic_rnn_seq2seq(
        X,
        decoder_inputs,
        cell
    )

    return dec_outputs


def singleLayer(X, outDim = 50, activation=tf.nn.relu):
    # Hidden layers
    head = tf.layers.flatten(X)
    # head = tf.layers.dense(head, outDim, activation=tf.nn.sigmoid, use_bias=True)
    head = tf.layers.dense(head, outDim, activation=activation, use_bias=True)
    return head


def dropout(X):
    return tf.layers.dropout(X)


def singleConvolution(X, numFilters = 5, filterSize=5, stride=3):
    # Hidden layers
    head = tf.layers.conv2d(X, numFilters, filterSize, stride, "same", activation=tf.nn.relu)
    return head


def trippleLayer(X, outDim = 128):
    head = tf.layers.flatten(X)
    head = tf.layers.dense(head, 64, activation=tf.nn.sigmoid, name="dense_1", use_bias=True)
    head = tf.layers.dense(head, 32, activation=tf.nn.sigmoid, name="dense_2", use_bias=True)
    head = tf.layers.dropout(head)
    head = tf.layers.dense(head, outDim, activation=None, name="dense_4", use_bias=True)
    return head

def fulllyConvPatchToPatch(X):
    head = X
    print('Input:', head.shape)
    head = tf.layers.conv2d(head, 20, 15, 5, "same", activation=tf.nn.relu)
    print('1:', head.shape)
    head = tf.layers.conv2d(head, 20, 10, 2, "same", activation=tf.nn.relu)
    print('2:', head.shape)
    head = tf.layers.conv2d(head, 40, 5, 1, "same", activation=tf.nn.relu)
    print('3:', head.shape)
    head = tf.layers.conv2d_transpose(head, 40, 5, 1, "same", activation=tf.nn.relu)  # 50 x 50
    print('4:', head.shape)
    head = tf.layers.conv2d_transpose(head, 20, 10, 2, "same", activation=tf.nn.sigmoid)  # 50 x 50
    print('5:', head.shape)
    head = tf.layers.conv2d_transpose(head, 20, 15, 5, "same", activation=None)  # 50 x 50
    print('Output:', head.shape)
    return head

def multiConvolution(X):
    head = X
    print('Input:', head.shape)
    head = tf.layers.conv2d(head, 10, 6, 3, "valid")  # 15 x 15
    print('1:', head.shape)
    head = tf.layers.conv2d(head, 20, 6, 1, "valid")  # 10 x 10
    print('2:', head.shape)
    head = tf.layers.conv2d_transpose(head, 10, 10, 4, "same")  # 40 x 40
    print('3:', head.shape)
    head = head[:,:,:-9,:] # 40 x 31
    print('4:', head.shape)
    head = tf.layers.conv2d(head, 64, 5, 1, "same")  # 40 x 31
    print('5:', head.shape)
    head = tf.layers.conv2d_transpose(head, 1, 15, 9, "same", activation=None)  # 360 x 274
    print('Output:', head.shape)
    return head

def reduceEnlarge(X):
    head = X
    print('Input:', head.shape)
    head = tf.layers.conv2d(head, 20, 7, 3, "same")
    print('1:', head.shape)
    head = tf.layers.conv2d(head, 17, 5, 3, "same")
    print('2:', head.shape)
    head = tf.layers.conv2d(head, 15, 3, 1, "same")
    print('2:', head.shape)
    head = tf.layers.conv2d_transpose(head, 15, 3, 1, "same")
    print('3:', head.shape)
    head = tf.layers.conv2d_transpose(head, 20, 5, 3, "same")
    print('3:', head.shape)
    head = tf.layers.conv2d_transpose(head, 20, 7, 3, "same")
    print('4:', head.shape)
    head = tf.layers.conv2d(head, 15, 5, 1, "same")
    print('5:', head.shape)
    head = tf.layers.conv2d_transpose(head, 1, 5, 1, "same", activation=None)
    print('Output:', head.shape)
    return head


#######################
train_batch =     100000
summary_step =     500
validation_step =   500
checkpoint_int =   20000
pre_train_steps =  500
save_pred_steps =  10000
#######################
use_split_pred = False
a = 0.0001  # GradNorm Weight
b = 0.00000000  # Prediction Weight
g = 0.005   # Scale for Phi
lr = 0.0005  # Learning Rate
#######################

########################################################################################################################

net_name = None # 'gru_predict_3_cells_200_low_lr'
saveDir = os.path.join('experiments', 'turbulence', 'recurrent_scaled_mse')


########################################################################################################################

def train(net_name=net_name, saveDir=saveDir, dataset_idx=LARGE_DATASET, loader=None, num_batches=train_batch,
          pixel_dropout=None):
    LOG_DIR = J('.', saveDir, net_name + '_' + datasets[dataset_idx] + '_lr' + str(lr))

    # Start beholder
    # beholder = Beholder(LOG_DIR)
    np.random.normal()

    # Load data
    pred_length = 200
    if loader is None:
        loader = Turbulence(pred_length=pred_length, dataset_idx=dataset_idx)
    else:
        pred_length = loader.pred_length

    # Load data onto GPU memory - ensure network layers have GPU support
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    # config.allow_soft_placement = True
    with tf.Session(config=config) as sess:

        # sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

        test = tf.placeholder_with_default(tf.constant(False, dtype=tf.bool), name="testing_flag", shape=())

        regions = tf.cond(test,
                          true_fn=lambda: loader.get_test_region_batch,
                          false_fn=lambda: loader.get_train_region_batch)

        # if True:
        with tf.device('/cpu:0'):
            with tf.name_scope('Data'):
                # Data does not fit into tensorflow data pipeline - so we split it later using a tensorflow slice op
                data = tf.constant(dtype=tf.float32, value=loader.get_data())

                # Get any input noise if present
                if loader.get_input_noise() is not None:
                    noise = tf.constant(dtype=tf.float32, value=loader.get_input_noise())
                    input_data = data + noise
                else:
                    input_data = data

                # Handle dropout if present
                if pixel_dropout is not None:
                    input_data = tf.layers.dropout(input_data, rate=pixel_dropout)

########################################################################################################################

            with tf.name_scope('Input'):
                # Simple 50 x 50 region with 20 frames of history.
                X = tf.map_fn(lambda region: tf.slice(input_data, region[0], [50, 50, 20]), regions, dtype=tf.float32)

                # Tensorflow wants time first, before transpose we have [batch, u, v, time]:[0, 1, 2, 3]
                X = tf.transpose(X, perm=[3, 0, 1, 2])

                # Ensure that the shape is well defined ( mapping slice operations is evaluated as a dynamic size)
                X = tf.reshape(X, (20, -1, 50 * 50))  # Flattened for fully connected layers

            with tf.name_scope('Label'):
                size = [50, 50, pred_length]
                Y = tf.map_fn(lambda region: tf.slice(data, region[1], size), regions, dtype=tf.float32)

                # Map the label to the same time first ordering
                Y = tf.transpose(Y, perm=[3, 0, 1, 2])

                outDim = [pred_length, -1, 50 * 50]
                # Y = tf.reshape(Y, outDim)

        print("Input shape:", X.shape)
        print("Output shape:", Y.shape)
        # print_op = tf.Print(X,[X])

########################################################################################################################

        #########################
        #     Define network    #
        #########################
#         with tf.name_scope('Encode'):

#             baseNetwork = lstm_encode(X, batchSize=loader.batch_size)
#             # baseNetwork = singleConvolution(X)
#             # baseNetwork = dropout(baseNetwork)
#             # baseNetwork = singleLayer(baseNetwork, outDim=32)
#             # baseNetwork = singleLayer(baseNetwork, outDim=64, activation=tf.nn.sigmoid)
#             # baseNetwork = multiConvolution(X)
#             # baseNetwork = reduceEnlarge(X)
#             # baseNetwork = seq2seq(X, outDim=512)  # [batch_size, seq_len, height, width]

#         with tf.name_scope('Decode'):
#             num_out = 50 * 50  # * loader.pred_length

#             state = lstm_decode(baseNetwork, batchSize=loader.batch_size, pred_length=pred_length)

#         with tf.name_scope('Prediction'):

#             # Map the enlarger network over the first axis (time)
#             Pred = tf.map_fn(lambda x: tf.layers.dense(x, num_out, activation=None), state)

#             # [200, 64, 50 * 50]              0,  1,  2,  3
#             Pred = tf.reshape(Pred, [pred_length, -1, 50, 50])  # Reshape the output to be width x height x 1
#             # [200, 64, 50, 50]
#             # Pred = tf.transpose(Pred, perm=[1, 2, 3, 0])
            
#         # [200, 64, 50 * 50]              0,  1,  2,  3
#         Pred = tf.reshape(Pred, [pred_length, -1, 50, 50])  # Reshape the output to be width x height x 1
        
        
            
        Pred = singleLayer(tf.transpose(X, perm=[1, 0, 2]), outDim=pred_length*50*50)
        Pred = tf.reshape(Pred, [-1, pred_length, 50, 50])  # Reshape the output to be width x height x 1
        Pred = tf.transpose(Pred, perm=[1, 0, 2, 3])



        # with tf.name_scope('Confidence'):
        #
        #     # Map the confidence estimation over the first axis (time)
        #     # [200, 64, 500]
        #     conf = tf.map_fn(lambda x: tf.layers.dense(x, 20, activation=tf.nn.elu), state)
        #     # [200, 64, 20]
        #     conf = tf.map_fn(lambda x: tf.layers.dense(x, 1, activation=tf.nn.elu), conf)
        #     # [200, 64, 1]
        #     conf = tf.reduce_mean(conf, axis=[1, 2])  # [200]
        #     # conf = tf.squeeze(conf)  # [200, 64]
        #
        #     avg_cong = tf.reduce_mean(conf)

########################################################################################################################

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
            # grads = adam.compute_gradients(conf_weighted_loss)
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
            tf.summary.image('Error', tf.expand_dims(Pred[0, :, :, :], axis=-1) - tf.expand_dims(Y[0, :, :, :], axis=-1), max_outputs=5),
            tf.summary.image('Mean Abs Error', tf.expand_dims(tf.reduce_mean(abs(tf.expand_dims(Pred[0, :, :, :], axis=-1) - tf.expand_dims(Y[0, :, :, :], axis=-1)), axis=0), axis=0), max_outputs=1),
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
                    # beholder.update(sess)

                if batch > pre_train_steps and batch % save_pred_steps == 0:
                    flags = dict({'testing_flag:0': True})
                    summaries, prediction, label, input_with_noise, accuracy = sess.run([merged_with_imgs, Pred, Y, X, pred_loss], feed_dict=flags)
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
