import os
import tensorflow as tf
from dataLoader.turbulence import Turbulence
import util.summaries
# from src.dataLoader.planets import get_data_

J = os.path.join
E = os.path.exists

## Architecures ##

def singleLayer(X, outDim = 50, name = False):
    # Hidden layers
    if name:
        head = tf.layers.dense(X, outDim, activation=tf.nn.sigmoid, use_bias=True, name="dense_1")
    else:
        head = tf.layers.dense(X, outDim, activation=tf.nn.sigmoid, use_bias=True)
    return head


def trippleLayer(X, outDim = 16):
    head = tf.layers.dense(X, 64, activation=tf.nn.sigmoid, name="dense_1", use_bias=True)
    head = tf.layers.dense(head, 24, activation=tf.nn.sigmoid, name="dense_2", use_bias=True)
    head = tf.layers.dense(head, outDim, activation=tf.nn.sigmoid, name="dense_3", use_bias=True)
    return head


#######################
train_epoch =   5000000
display_step =    10000
summary_step =     2000
checkpoint_int = 100000
pre_train_steps = 10000
#######################
use_split_pred = False
a = 0.0001  # GradNorm Weight
b = 0.00000000  # Prediction Weight
g = 0.005   # Scale for Phi
lr = 0.004  # Learning Rate
#######################

# saveDir = os.path.join('experiments', input("Name this run..."))
saveDir = os.path.join('experiments','test')

#%%mai
def main():

    # Load data
    loader = Turbulence()


    # Load data onto GPU memory - ensure network layers have GPU support
    with tf.Session() as sess:
        if True:
        # with tf.device('/gpu:0'):

            data = tf.constant(dtype=tf.float32, value=loader.get_data())

            test = tf.placeholder_with_default(tf.constant(False, dtype=tf.bool), name="testing_flag", shape=())

            regions = tf.cond(test, 
                true_fn=lambda: loader.get_test_region_batch,
                false_fn=lambda: loader.get_train_region_batch)

            X, Y = loader.map_regions(data, regions)

main()


#%%

            
            # sequence = dataLoader.pendulum.parallel_iterator(batch_size=128, sequence_len=5000)
            #
            # ic = sequence[0]
            # X = sequence[1]
            # F = sequence[2]

            # def train(elem):
            #     X = tf.stack(elem[0])
            #     F = tf.stack(elem[1])
            #     tf.print(X.get_shape())
            # Define placeholders
            # Xp = tf.placeholder(dtype= tf.float32, shape=[None, 4], name='Xp')

            #X = tf.placeholder(dtype= tf.float32, shape=[None, 2], name='X')
            #F = tf.placeholder(dtype= tf.float32, shape=[None, 2], name='F')

#             def calculate_loss(elem):

#                 X = tf.stack(elem[0])
#                 F = tf.stack(elem[1])

#                 if (b > 0):
#                     Y = tf.placeholder(dtype= tf.float32, shape=[None, 4], name='Y')

#                 # Define network
#                 with tf.name_scope('Base_Network'):
#                     if use_split_pred:
#                         with tf.variable_scope("auto_weight_sharing") as varScope:
#                             baseNetwork = singleLayer(X, outDim=32, name=True)
#                             varScope.reuse_variables()
#                             baseNetPrev = singleLayer(Xp, outDim=32, name=True)
#                     else:
#                         baseNetwork = singleLayer(X, outDim=32)

#                 with tf.name_scope('Phi'):
#                     Phi = singleLayer(baseNetwork, outDim = 1)

#                 if(b > 0):
#                     if use_split_pred:
#                         predNetwork = tf.concat([baseNetwork, baseNetPrev], axis=-1)
#                     else:
#                         predNetwork = baseNetwork
#                     with tf.name_scope('Prediction'):
#                         Pred = singleLayer(predNetwork, outDim=4)

#                 # Define Loss
#                 with tf.name_scope('gradPhi'):
#                     gradPhi = tf.gradients(Phi, [X])[0]

#                 # Calculate dot product
#                 with tf.name_scope('dotProd'):
#                     # dotProd = tf.reduce_sum(tf.multiply(F, gradPhi), axis=1)
#                     dotProd = tf.reduce_sum(tf.multiply(F, gradPhi)/tf.expand_dims(tf.norm(F, axis=1)*tf.norm(gradPhi, axis=1), dim=1), axis=1)

#                 # Calculate gradient regualization term
#                 with tf.name_scope('gradMag'):
#                     gradMag = tf.norm(gradPhi, axis=1)

#                 with tf.name_scope('grad_loss'):
#                     gradLoss = tf.reduce_mean(tf.square(gradMag - 1))
#                     #gradLoss = tf.reduce_mean(tf.maximum(gradMag - 2, 0) + tf.maximum(1 - gradMag, 0))

#                 if b > 0:
#                     with tf.name_scope('pred_loss'):
#                         predLoss = tf.losses.huber_loss(Y, Pred)

#                 with tf.name_scope('phi_mean'):
#                     mean = tf.reduce_mean(Phi)
#                     # phiLoss = tf.square(mean - 0.5) + tf.square(tf.sqrt(tf.reduce_mean(tf.square(Phi - mean))) - 0.08)
#                     phiLoss = tf.square(mean - 0.5)

#                 with tf.name_scope('loss'):
#                     alpha = tf.constant(a, dtype=tf.float32) # Scaling factor for magnitude of gradient
#                     beta  = tf.constant(b, dtype=tf.float32)  # Scaling factor for prediction of next time step
#                     gamma = tf.constant(g, dtype=tf.float32)  # Scaling factor for phi scale invariant
#                     loss = tf.reduce_mean(tf.abs(dotProd))

#                 if (a > 0):
#                     loss += alpha * gradLoss
#                 if (b > 0):
#                     loss += beta * predLoss
#                 if (g > 0):
#                     loss += gamma * phiLoss

#                 return loss
#         # with tf.name_scope('train'):
#         #     train_step = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss)

#         train_batch = tf.map_fn(lambda elem: tf.train.AdamOptimizer(learning_rate=lr).minimize(calculate_loss(elem)), [state_tensor, dynamic_tensor])


#         # TODO
#         # Use GPU to get a batch and then feed dict as we did before
#         # Give data loader a gpu session


#         # Create summary statistics outside of GPU scope
#         variable_summaries(Phi, "PhiSummary")
#         variable_summaries(gradPhi, "GradPhi")
#         variable_summaries(dotProd, "DotProduct")
#         variable_summaries(gradMag, "GradMagnitude")
#         tf.summary.scalar("GradLoss", gradLoss)
#         if b > 0:
#             tf.summary.scalar("PredictiveLoss", predLoss)
#         tf.summary.scalar("PhiLoss", phiLoss)
#         tf.summary.scalar("Cost", loss)

#         # Collect summary stats for train variables
#         merged = tf.summary.merge_all()

#         # Create checkpoint saver
#         saver = tf.train.Saver()

#         # Train the model

#         sess.run(tf.global_variables_initializer())

#         train_writer = tf.summary.FileWriter(
#             J('.', saveDir, 'train', 'alpha-' + str(a) + 'beta-' + str(b) + 'gama-' + str(g)), sess.graph)
#         # writers = [tf.summary.FileWriter(name) for name in summary_labels]
#         # dic = {'X:0': train_X, 'F:0': train_F, 'Y:0': train_Y}
#         # if b > 0:
#         #     dic['Xp:0'] = train_Xp
#         #     dic['Y:0'] = train_Y

#         for epoch in range(train_epoch + 1):
#             conds = dataLoader.pendulum.get_init(batch_size=128)
#             data = {initial_ph: conds}

#             # state, dynamic = sess.run([state_tensor, dynamic_tensor], feed_dict={initial_ph: conds})

#             # if epoch > pre_train_steps:
#             #     if epoch % summary_step == 0:
#             #         summary = sess.run([merged], feed_dict=data)[0]
#             #         train_writer.add_summary(summary, epoch)
#             #
#             #     if epoch % display_step == 0:
#             #         loss_ = sess.run(loss)
#             #         print(loss_, epoch)
#             #
#             #     if epoch % checkpoint_int == 0:
#             #         saver.save(sess, save_path=J('.', saveDir, 'network', str(epoch)))

#             sess.run([train_batch], feed_dict=data)



# main()