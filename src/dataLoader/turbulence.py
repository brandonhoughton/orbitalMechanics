#%%
import math
import os
import sys
import functools
import scipy.io as sio
import tensorflow as tf
import numpy as np


from pathlib import Path


#%%
print(os.getcwd())
from sklearn.model_selection import train_test_split

J = os.path.join
E = os.path.exists
dataDir = J('data', 'turbulence')
datasets = [
    'velocity_and_vorticity_field.mat',
    'velocity_and_vorticity_field_1200s.mat',
    'periodic_0',
    'periodic_1',
    'periodic_2',
    'periodic_3',
    'periodic_4']
SMALL_DATASET = 0
LARGE_DATASET = 1
TEST_DATASET_1 = 2
TEST_DATASET_2 = 3
TEST_DATASET_3 = 4
TEST_DATASET_4 = 5
TEST_DATASET_5 = 6

dynamics = 'vfield_local_dynamics.mat'
RANDOM_SEED = 42

#%%

# data = sio.loadmat(J(os.getcwd(), dataDir, datasets[LARGE_DATASET]))

# #%%
# data.keys()
# shape = data['U_t'].shape
# window_size = 50
# num_windows = 500

# low = [0 for _ in shape[:-1]]
# high = [x - window_size for x in shape[:-1]]

# np.random.seed(RANDOM_SEED)
# rectangles = tf.data.Dataset().from_tensor_slices(
#     np.random.uniform(low, high, size=(num_windows, len(high))))

# num_test = 100
# test_regions = rectangles.take(num_test).repeat()
# train_regions = rectangles.skip(num_test).repeat()


# getNext = test_slices.make_one_shot_iterator().get_next()

# with tf.Session() as ses:
#     for _ in range(2):
#         batch = ses.run(getNext)
#         print(batch)

#%%

#%%
def make_iterator(dataset, num_windows, batch_size):
    return dataset.shuffle(num_windows, RANDOM_SEED)\
            .batch(batch_size) \
            .prefetch(10) \
            .make_one_shot_iterator() \
            .get_next()


def make_iterator_no_shuffle(dataset, batch_size):
    return dataset.batch(batch_size) \
            .prefetch(1) \
            .make_one_shot_iterator() \
            .get_next()


def make_rectangles(location, window_size):
    new_location = location.copy()
    new_location[-1] += window_size[-1]

    return np.stack([location, new_location])


# Fist pass - just load 3d sections of turbulence data and train/test split them
# Sequence to frame model
class Turbulence:

    def __init__(self, batch_size=64, patch_size=50, window_size=20, num_windows=50000, pred_length=20,
                 dataset_idx=LARGE_DATASET, input_noise=None, debug=False):
        """

        :param batch_size: Number of samples per batch
        :param patch_size: Width and height of patch
        :param window_size: Number of frames of history given as input
        :param num_windows: Number of random patches
        :param pred_length: Number of predicted frames
        :param dataset_idx: Enum representing which dataset to load
        :param noise: Fixed noise array to add to input - must be the same shape as data
        :param debug: Whether or not to print shapes and sizes while loading the data
        """
        self.datasets = datasets
        self.noise = input_noise
        self.batch_size = batch_size
        # For small memory machines - just load the needed array rather than the whole .mat file
        self.data = sio.loadmat(J(os.getcwd(), dataDir, datasets[dataset_idx]))['U_t']
        self.shape = self.data.shape
        if debug:
            print('shape:', self.shape)

        self.pred_length = pred_length
        window_shape = [patch_size, patch_size, window_size]
        window_shape[-1] += self.pred_length
        if debug:
            print('window_shape', window_shape)

        # Define sub-sets of turbulent data
        low = [0 for _ in self.shape]
        high = [x - size for (x, size) in zip(self.shape, window_shape)]

        self.input_size = window_shape.copy()
        self.input_size[-1] -= self.pred_length
        self.test_size = [-1]
        self.test_size.extend(self.input_size.copy())
        self.test_size[-1] = self.pred_length
        self.num_out = 1
        self.num_test = batch_size

        if debug:
            print('x size', self.input_size)
            print('y size', self.test_size)

        for d in self.shape[:-1]:
            self.num_out *= abs(d)
        if debug:
            print("Number of parameter per sample of y:", self.num_out)

        np.random.seed(RANDOM_SEED)
        locations = np.random.uniform(low, high, size=(num_windows, len(high))).astype(np.int32)
        regions = np.apply_along_axis(
                    lambda location: make_rectangles(location, self.input_size), -1, locations)
        if debug:
            print("Example region:", regions[0])
        regions = tf.data.Dataset.from_tensor_slices(regions)

        # Make regions out of points
        # e.g. [[u, v, t],[width, height, length]]
        # regions = begins.map(lambda location: make_rectangles(location, window_size))

        # Train test split
        test_regions = regions.take(self.num_test).repeat()
        train_regions = regions.skip(self.num_test).repeat()

        self.get_test_region_batch = make_iterator_no_shuffle(test_regions, self.num_test)
        self.get_train_region_batch = make_iterator(train_regions, num_windows, self.batch_size)

        self.inputs = {
            'train_regions' : train_regions,    
            'test_regions'  : test_regions,
            'window_size'   : window_shape
        }

        # with tf.Session() as sess:
        #     for _ in range(20):
        #         a = sess.run(begins.make_one_shot_iterator().get_next())
        #         b = sess.run(regions.make_one_shot_iterator().get_next())
        #         c = sess.run(train_regions.make_one_shot_iterator().get_next())
        #         d = sess.run(self.get_train_region_batch)
        #         print(a)
        #         print(b)
        #         print(c)
        #         print(d)


    @staticmethod
    # Take a slice of a given tensor at a specific region
    # returns the entire time section of the region
    def slice_input(data: tf.Tensor, region):
        # X - input, slice at region (sequence)
        # return  tf.slice(data, region[0], region[1])
        return tf.slice(data, region[0], [50, 50, 10])

    @staticmethod
    def slice_label(data: tf.Tensor, region):
        # Y - result, slice of region at single timestep                 
        # return tf.slice(data,  region[2], region[3])
        return tf.slice(data,  region[2], [50, 50, 1])

    @staticmethod
    def map_regions(data: tf.Tensor, regions):
        return \
            tf.map_fn(lambda region: Turbulence.slice_input(data, region), regions, dtype=tf.float32), \
            tf.map_fn(lambda region: Turbulence.slice_label(data, region), regions, dtype=tf.float32)

    def get_data(self):
        # Scale data from [0, 1]
        dmin = np.min(self.data)
        dmax = np.max(self.data)
        return (self.data - dmin) / (dmax - dmin)
        # return self.data['U_t']

    def get_input_noise(self):
        return self.noise

    def get_train_regions(self, session):
        session.run(self.get_train_region_batch)

    def get_test_regions(self, session):
        session.run(self.get_test_region_batch)

# main()
# foo = Turbulence()

    # @staticmethod
    # def slice_label_(data : tf.Tensor, region):
    #     # Y - result, slice of region at single timestep
    #     indices = tf.constant([(0,2),(1,2)])
    #     startTime = region[0][-1] + region[1][-1]  # Move start of region to end of X time
    #     lengthTime = 1                             # Change size of time axis to be one frame
    #     updates = tf.constant(startTime, lengthTime)
    #     new_region = tf.scatter_nd_update(region, indices, updates)
    #     return tf.slice(data,  new_region[0], new_region[1])

    # def make_rectangles(begins, window_size):
    # x_size = window_size
    # x_size[-1] = window_size[-1] - 1
    # sizes = begins.map(lambda loc: np.array(x_size, dtype=np.int32))
    # return  {
    #     'begin': begins,
    #     'size' : sizes
    # }

