import os
import tensorflow as tf
from dataLoader.turbulence import Turbulence
import util.summaries

J = os.path.join
E = os.path.exists

## Architecures ##


def singleLayer(X, outDim = 50):
    # Hidden layers
    head = tf.layers.flatten(X)
    head = tf.layers.dropout(head)
    head = tf.layers.dense(head, outDim, activation=tf.nn.sigmoid, use_bias=True)
    return head

def singleConvolution(X, numFilters = 5, filterSize=10, stride=5):
    # Hidden layers
    head = tf.layers.conv2d(X, numFilters, filterSize, stride, "same")
    return head

def trippleLayer(X, outDim = 16):
    head = tf.layers.dense(X, 64, activation=tf.nn.sigmoid, name="dense_1", use_bias=True)
    head = tf.layers.dense(head, 24, activation=tf.nn.sigmoid, name="dense_2", use_bias=True)
    head = tf.layers.dense(head, outDim, activation=tf.nn.sigmoid, name="dense_3", use_bias=True)
    head = tf.layers.dropout(head)
    return head


#######################
train_batch =   50000000
summary_step =    10000
validation_step = 50000
checkpoint_int = 50000000
pre_train_steps = -100
#######################
use_split_pred = False
a = 0.0001  # GradNorm Weight
b = 0.00000000  # Prediction Weight
g = 0.005   # Scale for Phi
lr = 0.008  # Learning Rate
#######################

# saveDir = os.path.join('experiments', input("Name this run..."))
saveDir = os.path.join('experiments', 'turbulence')

#%%mai
def main():

    # Load data
    loader = Turbulence()

    # Load data onto GPU memory - ensure network layers have GPU support
    with tf.Session() as sess:
        if True:
        # with tf.device('/gpu:0'):

            # Data does not fit into tensorflow data pipeline - so we split it later using a tensorflow slice op
            data = tf.constant(dtype=tf.float32, value=loader.get_data())

            test = tf.placeholder_with_default(tf.constant(False, dtype=tf.bool), name="testing_flag", shape=())

            regions = tf.cond(test, 
                true_fn=lambda: loader.get_test_region_batch,
                false_fn=lambda: loader.get_train_region_batch)

            X, Y = loader.map_regions(data, regions)

            print(X.shape)
            # print_op = tf.Print(X,[X])

            # Define network
            with tf.name_scope('Base_Network'):
                baseNetwork = singleConvolution(X)

            predNetwork = baseNetwork
            with tf.name_scope('Prediction'):
                outDim = loader.test_size
                num_out = loader.num_out
                Pred = singleLayer(predNetwork, outDim=num_out)
                Pred = tf.reshape(Pred, outDim)  # Reshape the output to be width x height x 1( (may need batch size)

            with tf.name_scope('loss'):
                predLoss = tf.losses.huber_loss(Y, Pred)

        with tf.name_scope('train'):
            train_step = tf.train.AdamOptimizer(learning_rate=lr).minimize(predLoss)

        tf.summary.image('Predicted', Pred)
        tf.summary.image('Label', Y)
        tf.summary.scalar("PredictiveLoss", predLoss)

        # Collect summary stats for train variables
        merged = tf.summary.merge_all()

        # Create checkpoint saver
        saver = tf.train.Saver()

        sess.run(tf.global_variables_initializer())

        # sess.run(print_op)

        # Setup tensorboard logging directories
        train_writer = tf.summary.FileWriter(
            J('.', saveDir, 'turbulence_lr' + str(lr)), sess.graph)

        test_writer = tf.summary.FileWriter(
            J('.', saveDir, 'turbulence_lr' + str(lr)), sess.graph)

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
                summary = sess.run(merged, feed_dict=flags)
                test_writer.add_summary(summary, batch)

            if batch % checkpoint_int == 0:
                saver.save(sess, save_path=J('.', saveDir, 'network', str(batch)))

if __name__ == "__main__":
    main()
