import numpy as np
import math
import matplotlib.pyplot as plt
import tensorflow as tf
from mpl_toolkits.mplot3d import Axes3D

from dataLoader import get_data



with tf.Session() as sess:
  new_saver = tf.train.import_meta_graph('./network/500000.meta')
  new_saver.restore(sess, './network/500000')

  # Add input and infer.
  labels = tf.constant(0, tf.int32, shape=[100], name="labels")
  batch_size = tf.size(labels)
  phi = tf.get_collection("Phi")[0]
  loss = tf.losses.sparse_softmax_cross_entropy(labels=labels,
                                                logits=logits)

  tf.summary.scalar('loss', loss)
  # Creates the gradient descent optimizer with the given learning rate.
  optimizer = tf.train.GradientDescentOptimizer(0.01)

  # Runs train_op.
  train_op = optimizer.minimize(loss)
  sess.run(train_op)
