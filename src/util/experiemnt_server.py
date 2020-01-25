import os
import tensorflow as tf
import logging
import traceback
import threading


def spin_client_thread(hw_gpu_id=None):
    try:
        if hw_gpu_id is not None:
            os.environ['CUDA_VISIBLE_DEVICES'] = int(hw_gpu_id)

    except Exception as e:
        logging.error('Unable to start client thread:' + repr(e))


def brandon_cell(activation, skip_depth):
    # Conv layers
    num_filters = [8, 8, 32]
    kernel_size = [1, 3, 1]

    head1 = activation
    for i in range(skip_depth):
        if i != 0:
            head1 = tf.nn.relu(head1)
        head1 = tf.layers.conv2d(
            head1, num_filters[i], kernel_size[i], name='small_scale_layer_{}'.format(i), padding='same')

    # FC component
    head2 = activation
    head2 = tf.layers.conv2d(
        head2, filters=16, kernel_size=9, strides=5, name='big_scale_layer_1', activation=tf.nn.relu, padding='valid')
    head2 = tf.layers.conv2d(
        head2, filters=32, kernel_size=5, strides=1, name='big_scale_layer_2', activation=tf.nn.relu, padding='same')
    head2 = tf.layers.conv2d_transpose(
        head2, filters=32, kernel_size=10, strides=5, name='big_scale_layer_3', activation=tf.nn.relu, padding='valid')

    head = tf.concat([head1, head2], -1)

    head = tf.nn.relu(head + activation)

    # Skip layer
    return head


def boundry_cell(hidden_state, previous_state):
    #        next_state
    #            ^
    #  h ->  ######### -> h
    #            ^
    #      previous_state
    # Generate

def boundry_aware_recurrent(x, y, is_training):
    data = tf.random.normal(shape=[64, 20, 50, 50])

    h = tf.constant()


def multi_scale_concat_downsample(X, pred_len, is_training, skip_depth=3, encoder_kernel_size=1):
    # Input 50 x 50 X history_len patch
    # Map this patch to match the residual cell size
    head = X
    head = tf.layers.conv2d(
        head, filters=8, kernel_size=3, activation=tf.nn.relu, name='map_input_1',
        padding='same')
    head = tf.layers.conv2d(
        head, filters=8, kernel_size=3, activation=tf.nn.relu, name='map_input_2',
        padding='same')
    head = tf.layers.conv2d(
        head, filters=64, kernel_size=5, strides=2, activation=tf.nn.relu, name='map_input_3',
        padding='same')

    head = tf.layers.batch_normalization(head, training=is_training)

    head = tf.scan(
        fn=lambda acc, _: brandon_cell(acc, skip_depth),
        elems=tf.zeros(pred_len), initializer=head, swap_memory=True)

    head = tf.map_fn(
        fn=lambda elem:
        tf.layers.conv2d_transpose(inputs=elem, filters=8, kernel_size=5, strides=2, activation=tf.nn.relu,
                                   name='map_output_1',
                                   padding='same'),
        elems=head
    )
    head = tf.map_fn(
        fn=lambda elem:
        tf.layers.conv2d(inputs=elem, filters=1, kernel_size=3, name='map_output_2', padding='same'),
        elems=head
    )
    head = tf.squeeze(head)

    return head