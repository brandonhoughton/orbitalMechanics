import os
import tensorflow as tf
from tensorflow import keras
from dataLoader.turbulence import Turbulence
import util.summaries

J = os.path.join
E = os.path.exists

## Architecures ##


def singleLayer(X, outDim = 50):
    # Hidden layers
    head = tf.layers.flatten(X)
    head = tf.layers.dropout(head)
    # head = tf.layers.dense(head, outDim, activation=tf.nn.sigmoid, use_bias=True)
    head = tf.layers.dense(head, outDim, activation=None, use_bias=True)
    return head


def singleConvolution(X, numFilters = 5, filterSize=10, stride=5):
    # Hidden layers
    head = tf.layers.conv2d(X, numFilters, filterSize, stride, "same")
    return head


def trippleLayer(X, outDim = 16):
    head = tf.layers.flatten(X)
    head = tf.layers.dense(head, 64, activation=tf.nn.sigmoid, name="dense_1", use_bias=True)
    head = tf.layers.dropout(head)
    head = tf.layers.dense(head, 12, activation=tf.nn.sigmoid, name="dense_2", use_bias=True)
    head = tf.layers.dense(head, outDim, activation=None, name="dense_3", use_bias=True)
    return head

def multiConvolution(X):
    head = X
    print('Input:', head.shape)
    head = tf.layers.conv2d(head, 10, 6, 3, "valid") # 15 x 15
    print('1:', head.shape)
    head = tf.layers.conv2d(head, 20, 6, 1, "valid") # 10 x 10
    print('2:', head.shape)
    head = tf.layers.conv2d_transpose(head, 10, 10, 4, "same") # 40 x 40
    print('3:', head.shape)
    head = head[:,:,:-9,:] # 40 x 31
    print('4:', head.shape)
    head = tf.layers.conv2d(head, 64, 5, 1, "same") # 40 x 31
    print('5:', head.shape)
    head = tf.layers.conv2d_transpose(head, 1, 15, 9, "same", activation=None) # 360 x 274
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
train_batch =   1000000000
summary_step =    20000000
validation_step = 20000000
checkpoint_int = 5000000000
pre_train_steps = -100
#######################
use_split_pred = False
a = 0.0001  # GradNorm Weight
b = 0.00000000  # Prediction Weight
g = 0.005   # Scale for Phi
lr = 0.005  # Learning Rate
#######################

# saveDir = os.path.join('experiments', input("Name this run..."))
saveDir = os.path.join('experiments', 'turbulence')

#%%mai
def main():

    # Load data
    loader = Turbulence()

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
            #X = tf.map_fn(lambda region: tf.slice(data, region[0], [50, 50, 20]), regions, dtype=tf.float32)

            # Padded region with 0's everywhere except for where the patch is e.g.
            #20 *  ########
                   #      #
                   #      #
                   #  xx  #
                   #  xx  #
                   ########
            X = tf.map_fn(lambda region:
                          tf.pad(
                              tf.slice(data, region[0], [50, 50, 20]),
                              [[region[0, 0], loader.shape[0] - region[0, 0] - 50], [region[0, 1], loader.shape[1] - region[0, 1] - 50], [0, 0]],
                              "CONSTANT"),
                          regions,
                          dtype=tf.float32,
                          parallel_iterations=12)
            X = tf.reshape(X, (-1, 360, 279, 20))
                    


            # Complete region 1 frame in the future
            size = [loader.shape[0], loader.shape[1], 1]
            Y = tf.map_fn(lambda region: tf.slice(data, region[2], size), regions, dtype=tf.float32)


        print(X.shape)
        # print_op = tf.Print(X,[X])

        # Define network
        with tf.name_scope('Base_Network'):
            # baseNetwork = singleConvolution(X)
            # baseNetwork = multiConvolution(X)
            baseNetwork = reduceEnlarge(X)

        predNetwork = baseNetwork
        with tf.name_scope('Prediction'):
            # outDim = [-1]
            # outDim.extend(size)
            # num_out = loader.num_out
            # Pred = trippleLayer(predNetwork, outDim=num_out)
            # Pred = tf.reshape(Pred, outDim)  # Reshape the output to be width x height x 1( (may need batch size)
            Pred = predNetwork

        with tf.name_scope('loss'):
            predLoss = tf.losses.huber_loss(Y, Pred)
            # predLoss = tf.losses.mean_squared_error(Y, Pred)

        with tf.name_scope('train'):
            train_step = tf.train.AdamOptimizer(learning_rate=lr).minimize(predLoss)

        tf.summary.scalar("PredictiveLoss", predLoss)
        # tf.summary.histogram("loss", Pred - Y)

        # Collect summary stats for train variables
        merged = tf.summary.merge_all()

        marged_with_imgs = \
            [merged,
            tf.summary.image('Predicted', Pred, max_outputs=5),
            tf.summary.image('Label', Y, max_outputs=5),
            tf.summary.image('Error', Pred - Y, max_outputs=5),
            tf.summary.image('Mean Abs Error', tf.expand_dims(tf.reduce_mean(abs(Pred - Y), axis=0), axis=0), max_outputs=1)]

        # Create checkpoint saver
        saver = tf.train.Saver()

        sess.run(tf.global_variables_initializer())

        # sess.run(print_op)

        # Setup tensorboard logging directories
        net_name = 'padded_multi_conv_pred_full'
        train_writer = tf.summary.FileWriter(
            J('.', saveDir, net_name + '_lr' + str(lr), 'train'), sess.graph)

        test_writer = tf.summary.FileWriter(
            J('.', saveDir, net_name + '_lr' + str(lr), 'validation'), sess.graph)

        # Train model
        for batch in range(train_batch + 1):
            if batch > pre_train_steps:
                if batch % summary_step == 0:
                    loss, _, summary = sess.run([predLoss, train_step, merged])
                    train_writer.add_summary(summary, batch)
                    print(loss, batch)
            else:
                sess.run([predLoss, train_step, merged])

            if batch % validation_step == 0:
                flags = dict({'testing_flag:0':True})
                summaries = sess.run(marged_with_imgs, feed_dict=flags)
                for summary in summaries:
                    test_writer.add_summary(summary, batch)

            if batch % checkpoint_int == 0:
                saver.save(sess, save_path=J('.', saveDir, 'network', str(batch)))

if __name__ == "__main__":
    main()
