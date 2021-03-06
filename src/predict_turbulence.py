import os
import tensorflow as tf
from tensorflow import keras
from tensorboard.plugins.beholder import Beholder
from dataLoader.turbulence import Turbulence
import util.summaries


J = os.path.join
E = os.path.exists

## Architecures ##


def lstm(X, outDim = 50):
    rnn_cell = tf.nn.rnn_cell.BasicLSTMCell(outDim)
    initial_state = rnn_cell.zero_state(200, dtype=tf.float32)
    head, _ = tf.nn.dynamic_rnn(rnn_cell, X, initial_state=initial_state, dtype=tf.float32)
    return head

def seq2seq(X, outDim= 512, outLen=10, num_layers=3):
    # Create a `layers_stacked_count` of stacked RNNs (GRU cells here).
    cells = []
    for i in range(num_layers):
        with tf.variable_scope('RNN_{}'.format(i)):
            cells.append(tf.nn.rnn_cell.GRUCell(outDim))
            # cells.append(tf.nn.rnn_cell.BasicLSTMCell(...))
    cell = tf.nn.rnn_cell.MultiRNNCell(cells)
    decoder_inputs = [ tf.zeros_like(X[0], dtype=tf.float32, name="GO")] + X[:-1]
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
train_batch =   10000000
summary_step =      2000
validation_step =   2000
checkpoint_int = 500000
pre_train_steps = -100
#######################
use_split_pred = False
a = 0.0001  # GradNorm Weight
b = 0.00000000  # Prediction Weight
g = 0.005   # Scale for Phi
lr = 0.004  # Learning Rate
#######################

# saveDir = os.path.join('experiments', input("Name this run..."))
saveDir = os.path.join('experiments', 'turbulence')

net_name = 'sin_func_test'

LOG_DIR = J('.', saveDir, net_name + '_lr' + str(lr))

def main():

    # Start beholder
    beholder = Beholder(LOG_DIR)

    # Load data
    loader = Turbulence(pred_length=20)

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
            # Data does not fit into tensorflow data pipeline - so we split it later using a tensorflow slice op
            data = tf.constant(dtype=tf.float32, value=loader.get_data())
            
            # Simple 50 x 50 region with 20 frames of history. e.g. 20 x 
            # 20 * ####
                   #xx#
                   #xx#
                   ####
            # X = tf.map_fn(lambda region: tf.slice(data, region[0], [50, 50, 20]), regions, dtype=tf.float32)
            X = tf.cast(regions[:, 0], dtype=tf.float32)

            #For time series
            # X = tf.transpose(X, perm=[0, 3, 1, 2])
            # X = tf.reshape(X, (-1, 20, 50, 50))

            # Padded region with 0's everywhere except for where the patch is e.g.
            #20 *  ########
                   #      #
                   #      #
                   #  xx  #
                   #  xx  #
            #        ########
            # X = tf.map_fn(lambda region:
            #               tf.pad(
            #                   tf.slice(data, region[0], [50, 50, 20]),
            #                   [[region[0, 0], loader.shape[0] - region[0, 0] - 50], [region[0, 1], loader.shape[1] - region[0, 1] - 50], [0, 0]],
            #                   "CONSTANT"),
            #               regions,
            #               dtype=tf.float32,
            #               parallel_iterations=12)
            # X = tf.reshape(X, (-1, 360, 279, 20))



            # Complete region 1 frame in the future
            size = [50, 50, 20]
            # size = [loader.shape[0], loader.shape[1], loader.pred_length]
            Y = tf.map_fn(lambda region: tf.slice(data, region[0], size), regions, dtype=tf.float32)
            # size = [loader.pred_length, loader.shape[0], loader.shape[1]]
            # Y = tf.transpose(Y, perm=[0, 3, 1, 2])


        print(X.shape)
        # print_op = tf.Print(X,[X])

        # Define network
        with tf.name_scope('Base_Network'):
            baseNetwork = trippleLayer(X)
            # baseNetwork = singleConvolution(X)
            # baseNetwork = dropout(baseNetwork)
            # baseNetwork = singleLayer(baseNetwork, outDim=32)
            # baseNetwork = singleLayer(baseNetwork, outDim=64, activation=tf.nn.sigmoid)
            # baseNetwork = multiConvolution(X)
            # baseNetwork = reduceEnlarge(X)
            # baseNetwork = seq2seq(X, outDim=512)  # [batch_size, seq_len, height, width]

        predNetwork = baseNetwork
        with tf.name_scope('Prediction'):
            outDim = [-1]
            outDim.extend(size)
            num_out = 50 * 50 * loader.pred_length #loader.num_out
            Pred = singleLayer(baseNetwork, outDim=num_out, activation=None)
            # Pred = baseNetwork
            Pred = tf.reshape(Pred, outDim)  # Reshape the output to be width x height x 1( (may need batch size)
            # Pred = predNetwork

        with tf.name_scope('loss'):
            predLoss = tf.losses.huber_loss(Y, Pred)
            # predLoss = tf.losses.mean_squared_error(Y, Pred)

        with tf.name_scope('train'):
            adam = tf.train.AdamOptimizer(learning_rate=lr)
            grads = adam.compute_gradients(predLoss)
            train_step = adam.apply_gradients(grads)

        tf.summary.scalar("PredictiveLoss", predLoss)
        for grad in grads:
            tf.summary.scalar("NormGradL1" + str(grad), tf.reduce_mean(tf.abs(grad)))
            tf.summary.histogram("grad" + str(grad), grad)

        # Collect summary stats for train variables
        merged = tf.summary.merge_all()

        marged_with_imgs = \
            [merged,
            tf.summary.image('Predicted_t0', Pred[:, :, :, 0:1], max_outputs=5),
            tf.summary.image('Label', Y[:, :, :, 0:1], max_outputs=5),
            tf.summary.image('Error', Pred[:, :, :, 0:1] - Y[:, :, :, 0:1], max_outputs=5),
            tf.summary.image('Mean Abs Error', tf.expand_dims(tf.reduce_mean(abs(Pred[:, :, :, 0:1] - Y[:, :, :, 0:1]), axis=0), axis=0), max_outputs=1),
             ##
            tf.summary.image('Predicted_t5', Pred[:, :, :, 5:6], max_outputs=5),
            tf.summary.image('Label_t5', Y[:, :, :, 5:6], max_outputs=5),
            tf.summary.image('Error_t5', Pred[:, :, :, 5:6] - Y[:, :, :, 5:6], max_outputs=5),
            tf.summary.image('Mean Abs Error_t5', tf.expand_dims(tf.reduce_mean(abs(Pred[:, :, :, 5:6] - Y[:, :, :, 5:6]), axis=0), axis=0), max_outputs=1),
             ##
            tf.summary.image('Predicted_t10', Pred[:, :, :, 10:11], max_outputs=5),
            tf.summary.image('Label_t10', Y[:, :, :, 10:11], max_outputs=5),
            tf.summary.image('Error_t10', Pred[:, :, :, 10:11] - Y[:, :, :, 10:11], max_outputs=5),
            tf.summary.image('Mean Abs Error_t10', tf.expand_dims(tf.reduce_mean(abs(Pred[:, :, :, 10:11] - Y[:, :, :, 10:11]), axis=0), axis=0), max_outputs=1),
             ##
            tf.summary.image('Predicted_t15', Pred[:, :, :, 15:16], max_outputs=5),
            tf.summary.image('Label_t15', Y[:, :, :, 15:16], max_outputs=5),
            tf.summary.image('Error_t15', Pred[:, :, :, 15:16] - Y[:, :, :, 15:16], max_outputs=5),
            tf.summary.image('Mean Abs Error_t15', tf.expand_dims(tf.reduce_mean(abs(Pred[:, :, :, 15:16] - Y[:, :, :, 15:16]), axis=0), axis=0), max_outputs=1),
            ##
            tf.summary.image('Predicted_t19', Pred[:, :, :, 19:20], max_outputs=5),
            tf.summary.image('Label_t19', Y[:, :, :, 19:20], max_outputs=5),
            tf.summary.image('Error_t19', Pred[:, :, :, 19:20] - Y[:, :, :, 19:20], max_outputs=5),
            tf.summary.image('Mean Abs Error_t19', tf.expand_dims(tf.reduce_mean(abs(Pred[:, :, :, 19:20] - Y[:, :, :, 19:20]), axis=0), axis=0), max_outputs=1)]

        # Create checkpoint saver
        saver = tf.train.Saver()

        sess.run(tf.global_variables_initializer())

        # sess.run(print_op)

        # Setup tensorboard logging directories
        train_writer = tf.summary.FileWriter(
            J(LOG_DIR, 'train'), sess.graph)

        test_writer = tf.summary.FileWriter(
            J(LOG_DIR, 'validation'), sess.graph)

        # Train model
        for batch in range(train_batch + 1):
            if batch > pre_train_steps and batch % summary_step == 0:
                loss, summary = sess.run([predLoss, merged])
                train_writer.add_summary(summary, batch)
                print(loss, batch)
            else:
                sess.run(train_step)
                # beholder.update(sess)

            if batch % validation_step == 0:
                flags = dict({'testing_flag:0': True})
                summaries = sess.run(marged_with_imgs, feed_dict=flags)
                for summary in summaries:
                    test_writer.add_summary(summary, batch)

            if batch % checkpoint_int == 0:
                saver.save(sess, save_path=J(LOG_DIR, 'network', str(batch)))

if __name__ == "__main__":
    main()
