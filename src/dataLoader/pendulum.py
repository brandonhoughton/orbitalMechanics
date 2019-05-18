import functools

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from scipy.integrate import odeint
import tqdm


def RK4(f, y0, tmax, h=0.1):
    """
    Implements the Rk4 method."""
    with tf.variable_scope("RK4"):
        y = [y0]
        f_overt = []
        tint = np.linspace(h, tmax, tmax / h)
        # h - > tmax, with step sizes tmax/h
        for tn in tqdm.tqdm(tint[1:]):
            with tf.variable_scope("step_{}".format(tn)):
                yn = y[-1]

                with tf.variable_scope("k1"):
                    k1 = f(tn, yn)
                with tf.variable_scope("k2"):
                    k2 = f(tn + h / 2, yn + h / 2 * k1)
                with tf.variable_scope("k3"):
                    k3 = f(tn + h / 2, yn + h / 2 * k2)
                with tf.variable_scope("k4"):
                    k4 = f(tn + h, yn + h * k3)

                y.append(yn + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4))
                f_overt.append(f(tn, yn))

    return tint, y[1:], f_overt  # time line, position, derivative



def pendulum(length, g=9.81):
    """
    Represents the pendulum differential equation.
    """
    with tf.variable_scope("pend"):
        def _dif_eq(t, state):
            theta = state[:, 0]
            dtheta = state[:, 1]

            out = tf.stack([dtheta, -g / length * tf.sin(theta)], axis=-1)
            return out
    return _dif_eq


def get_tensors(sess):
    initial_ph = tf.placeholder(tf.float32, shape=[None, 2])
    pfunc = pendulum(1)
    tint, state_tensor, dynamic_tensor = RK4(pfunc, initial_ph, 5, h=0.1)

    return initial_ph, state_tensor, dynamic_tensor


def get_init(batch_size):
    return np.stack(
        [np.random.uniform(-np.pi, np.pi, (batch_size)),
         np.zeros(batch_size)], axis=-1)


def test():
    sess = tf.InteractiveSession()

    initial_ph, state_tensor, dynamic_tensor = get_tensors(sess)

    conds = get_init(batch_size=128)
    print(conds.shape)

    state, dynamic = sess.run([state_tensor, dynamic_tensor], feed_dict={initial_ph: conds})

    import matplotlib.pyplot as plt
    state = np.array(state)
    # for s in state:
    for i in range((state.shape[1])):
        plt.plot(state[:, i, 0])
    #     break
    plt.xlabel("time")
    plt.ylabel("theta")
    plt.show()


def ODE(qp, _):
    return qp[1], -np.sin(qp[0])


def Htrue(q, p):
    return p ** 2 / 2 + (1 - np.cos(q))


def simulate(ic, T_values, backward=False):
    print(ic.shape)
    ic = np.random.uniform(low=-1, high=1, size=2)
    # print(T_values.shape, T_values)
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


class Pend:
    def __init__(self, length=500, high=None):
        self.current = 0
        self.high = high
        self.len = length
        self.step = 0.1

    def __iter__(self):
        return self

    def __next__(self):
        if self.high is not None and self.current > self.high:
            raise StopIteration
        else:
            self.current += 1
            time_steps = np.linspace(start=self.step, stop=self.len * self.step, num=self.len)
            print(len(time_steps))
            return simulate(np.random.uniform(low=-1, high=1, size=2), T_values=time_steps)





class PendSeed:
    def __init__(self, high=None):
        self.current = 0
        self.high = high

    def __iter__(self):
        return self

    def __next__(self):
        if self.high is not None and self.current > self.high:
            raise StopIteration
        else:
            self.current += 1
            return np.array([self.current])

    @staticmethod
    def simulate_seed(elem, seq_len=2):
        elem = elem
        time_steps = np.linspace(start=0.1, stop=seq_len * 0.1, num=seq_len)
        fghfgthfghfgh = np.random.uniform(low=-1, high=1, size=2)
        print(fghfgthfghfgh)
        return simulate(fghfgthfghfgh, T_values=time_steps)


def simulate_seed(elem, seq_len=2):
    print("I was called")
    return elem
    return np.random.uniform(low=-1, high=1, size=2)
    time_steps = np.linspace(start=0.1, stop=seq_len * 0.1, num=seq_len)
    fghfgthfghfgh = np.random.uniform(low=-1, high=1, size=2)
    print(fghfgthfghfgh)
    return simulate(fghfgthfghfgh, T_values=time_steps)


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
        .make_one_shot_iterator().get_next()


def parallel_iterator(sequence_len, batch_size, num_cpu=8, num_rep=100):
    # simFunc = functools.partial(PendSeed.simulate_seed, sequence_len)
    return tf.data.Dataset().from_generator(
        PendSeed,
        output_types=tf.int32,
        output_shapes=(tf.TensorShape([1]))) \
        .map(simulate_seed, num_parallel_calls=num_cpu) \
        .shuffle(batch_size * batch_size) \
        .make_one_shot_iterator().get_next()

def tensor_iterator(sequence_len, batch_size, num_cpu=8, num_rep=100):
    # simFunc = functools.partial(PendSeed.simulate_seed, sequence_len)
    return tf.data.Dataset().from_tensor_slices(
        PendSeed,
        output_types=tf.int32,
        output_shapes=(tf.TensorShape([1]))) \
        .map(simulate_seed, num_parallel_calls=num_cpu) \
        .shuffle(batch_size * batch_size) \
        .make_one_shot_iterator().get_next()


# Test dataset
# this form will make it easy to sample batches of arbitrary sequence length
if __name__ == "__main__":
    test()

    el = parallel_iterator(batch_size=32, sequence_len=2, num_rep=1)
    with tf.Session() as sess:
        ic = []
        x = []
        f = []
        for _ in range(20):
            batch = sess.run(el)
            ic.append(batch[0])
            # x.append(batch[1])
            # f.append(batch[2])
        print(ic)
        print(x)

        plt.scatter(x[:,0], x[:,1], alpha=0.5)

        plt.show()

        plt.scatter(f[:, 0], f[:, 1], alpha=0.5)

        plt.show()
