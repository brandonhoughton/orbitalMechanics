import numpy as np
import tensorflow as tf
from scipy.integrate import odeint

def ODE(qp, _):
    return qp[1],   -np.sin(qp[0])
 
def Htrue(q, p):
    return p ** 2 / 2 + (1 - np.cos(q))
 
def simulate(ic, T_values=None, backward=False):
    print(ic.shape, ic)
    print(T_values.shape, T_values)
    if T_values is None: T_values = np.linspace(0, 20, 100)
    if backward:
        f = lambda x, t: ODE(-x, t)
    else:
        f = ODE
    Y = odeint(f, ic, T_values)
    F = [f(y, t) for t, y in zip(T_values, Y)]
    return np.stack([
        np.tile(ic, (len(T_values), 1)),
        F,
        odeint(f, ic, T_values)], axis=0).astype(np.float32)
    # return np.tile(ic, (len(T_values), 1)), F, odeint(f, ic, T_values)
    # return ic, F, odeint(f, ic, T_values)


class Pend:
    def __init__(self, length=500, high=None):
        self.current = 0
        self.high = high
        self.len = length
        self.step = 1

    def __iter__(self):
        return self

    def __next__(self): 
        if self.high is not None and self.current > self.high:
            raise StopIteration
        else:
            self.current += 1
            time_steps = np.linspace(start = self.step, stop = self.len * self.step, num=self.len)
            print(len(time_steps))
            return simulate(np.random.uniform(low=-1, high=1, size=2), T_values=time_steps)


class PendSeed:
    def __init__(self, high=None):
        self.current = 0
        self.high = high
        self.step = 1

    def __iter__(self):
        return self

    def __next__(self):
        if self.high is not None and self.current > self.high:
            raise StopIteration
        else:
            self.current += 1
            return np.array([self.current])

    @staticmethod
    def simulate_seed(elem):
        seq_len = 500
        time_steps = np.linspace(start=10, stop=seq_len * 10, num=seq_len)
        return simulate(np.random.uniform(low=-1, high=1, size=2), T_values=time_steps)



def dat():
    return tf.data.Dataset().from_generator(
        Pend,
        output_types=tf.float32,
        output_shapes=(tf.TensorShape([3, None, 2])))


def iterator(sequence_len, batch_size):
    return tf.data.Dataset().from_generator(
        Pend,
        output_types=tf.float32,
        output_shapes=(tf.TensorShape([3, sequence_len, 2])),
        args=[sequence_len]) \
        .shuffle(2*batch_size) \
        .prefetch(2) \
        .make_one_shot_iterator().get_next()


def parallel_iterator(sequence_len, batch_size, num_cpu=8):
    return tf.data.Dataset().from_generator(
        PendSeed,
        output_types=tf.int32,
        output_shapes=(tf.TensorShape(1))) \
        .map(PendSeed.simulate_seed, num_parallel_calls=num_cpu) \
        .interleave(lambda x: tf.data.Dataset.from_tensors(x).repeat(6), cycle_length=batch_size, block_length=2) \
        .prefetch(batch_size) \
        .make_one_shot_iterator().get_next()


# Test dataset
# this form will make it easy to sample batches of arbitrary sequence length
# el = iterator(50, 32)
# with tf.Session() as sess:
#     print(sess.run(el[0]))
