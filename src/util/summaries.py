import tensorflow as tf

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