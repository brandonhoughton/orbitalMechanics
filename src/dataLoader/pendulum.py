import numpy as np
import tensorflow as tf
from scipy.integrate import odeint

def ODE(qp, _):
    return qp[1],   -np.sin(qp[0])
 
def Htrue(q, p):
    return p ** 2 / 2 + (1 - np.cos(q))
 
def simulate(ic, T_values=None, backward=False):
    if T_values is None: T_values = np.linspace(0, 20, 100)
    if backward:
        f = lambda x, t: ODE(-x, t)
    else:
        f = ODE
    Y = odeint(f, ic, T_values)
    F = [f(y, t) for t, y in zip(T_values, Y)]
    return ic, F[1], odeint(f, ic, T_values)[1]


class Pend:
    def __init__(self, high = None):
        self.current = 0
        self.high = high

    def __iter__(self):
        return self

    def __next__(self): 
        if self.high is not None and self.current > self.high:
            raise StopIteration
        else:
            self.current += 1
            return simulate(np.random.uniform(low=-1, high=1, size=2), T_values=np.random.uniform(low=0, high=200, size=2))


# Test dataset
# this form will make it easy to sample batches of arbitrary sequence length
dataset = tf.data.Dataset().from_generator(
    Pend,
    output_types=tf.float32,
    output_shapes=(tf.TensorShape([2, 2])))

# iter = dataset.make_initializable_iterator()
iter = dataset.batch(batch_size=3).make_one_shot_iterator()
el = iter.get_next()

with tf.Session() as sess:
    # sess.run(iter.initializer)
    print(sess.run(el))
    print(sess.run(el))
    print(sess.run(el))
