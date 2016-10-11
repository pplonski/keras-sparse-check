from scipy import sparse

import time
import threading

import numpy as np
np.random.seed(1337)  # for reproducibility
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.utils import np_utils


# training parameters
batch_size = 128
nb_classes = 3
nb_epoch = 2
# input image dimensions
img_rows, img_cols = 32, 32
# number of convolutional filters to use
nb_filters = 64
# size of pooling area for max pooling
nb_pool = 2
# convolution kernel size
nb_conv = 3

# input files with dense matrices
input_files = [ '/media/2t/CNN_DATA/subset_db_x_view_1_1000.npy', 
                    '/media/2t/CNN_DATA/subset_db_x_view_1_2000.npy',
                    '/media/2t/CNN_DATA/subset_db_x_view_1_3000.npy',
                    '/media/2t/CNN_DATA/subset_db_x_view_1_4000.npy',
                    '/media/2t/CNN_DATA/subset_db_x_view_1_5000.npy',
                    '/media/2t/CNN_DATA/subset_db_x_view_1_6000.npy',
                    '/media/2t/CNN_DATA/subset_db_x_view_1_7000.npy',]

y_input_files = ['/media/2t/CNN_DATA/subset_db_y_view_1_1000.npy', 
                 '/media/2t/CNN_DATA/subset_db_y_view_1_2000.npy',
                 '/media/2t/CNN_DATA/subset_db_y_view_1_3000.npy',
                 '/media/2t/CNN_DATA/subset_db_y_view_1_4000.npy',
                 '/media/2t/CNN_DATA/subset_db_y_view_1_5000.npy',
                 '/media/2t/CNN_DATA/subset_db_y_view_1_6000.npy',
                 '/media/2t/CNN_DATA/subset_db_y_view_1_7000.npy']

# Keras network (simple architecture)
def get_model():
    model = Sequential()
    model.add(Convolution2D(nb_filters, nb_conv, nb_conv,
                            border_mode='valid',
                            input_shape=(1, img_rows, img_cols)))
    model.add(Activation('relu'))
    model.add(Convolution2D(nb_filters, nb_conv, nb_conv))
    model.add(Activation('relu'))

    model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
    model.add(Flatten())
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adadelta')
    return model

# return Y data
def get_y_data():
    
    y = None
    for file_name in y_input_files:
        print 'Reading from', file_name        
        tmp = np.load(file_name)
        y = tmp if y is None else np.concatenate((y, tmp)) 
        del tmp
    print 'Y shape', y.shape
    return y

# return X data as list of sparse metrices
def get_x_data_sparse():
    # array of sparse matrices 
    data = []
    for file_name in input_files:
        print 'Reading from', file_name
        # temporary load as dense
        tmp = np.load(file_name)
        for i in xrange(0, tmp.shape[0]):
            data += [sparse.csr_matrix(tmp[i])]
        del tmp
    print 'X samples', len(data)
    return data

# return X data as dense numpy array
def get_x_data_dense():
    # array of sparse matrices 
    data = None
    for file_name in input_files:
        print 'Reading from', file_name
        # temporary load as dense
        tmp = np.load(file_name)
        data = tmp if data is None else np.concatenate((data, tmp))    
        del tmp
    data = data.reshape((data.shape[0], 1, data.shape[1], data.shape[2]))
    print 'X shape', data.shape
    return data


def split_data(X, y):
    # how to split data between training and validation
    split_on = 2000000
    X_train = X[:split_on]
    X_test  = X[split_on:]
    Y_train = y[:split_on]
    Y_test  = y[split_on:] 
    return X_train, Y_train, X_test, Y_test

#
# Generator
#
class threadsafe_iter:
    """Takes an iterator/generator and makes it thread-safe by
    serializing call to the `next` method of given iterator/generator.
    """
    def __init__(self, it):
        self.it = it
        self.lock = threading.Lock()

    def __iter__(self):
        return self

    def next(self):
        with self.lock:
            return self.it.next()


def threadsafe_generator(f):
    """A decorator that takes a generator function and makes it thread-safe.
    """
    def g(*a, **kw):
        return threadsafe_iter(f(*a, **kw))
    return g


@threadsafe_generator
def sparse_generator(X, Y, batch_size=128, shuffle=True):
    number_of_batches = np.ceil(len(X)/batch_size)
    sample_index = np.arange(len(X))
    if shuffle:
        np.random.shuffle(sample_index)
    img_rows = X[0].shape[0]
    img_cols = X[0].shape[1]
    nb_classes = Y.shape[1]
    
    counter = 0
    while True:
        batch_index = sample_index[batch_size*counter:min(batch_size*(counter+1), len(X))]
        X_batch = np.zeros((len(batch_index), 1, img_rows, img_cols))
        y_batch = np.zeros((len(batch_index), nb_classes))
        
        for i, j in enumerate(batch_index):
            X_batch[i,0,:,:] = X[j].toarray()
            y_batch[i] = Y[j]      
            
        counter += 1
        yield X_batch, y_batch
        if counter == number_of_batches:
            if shuffle:
                np.random.shuffle(sample_index)
            counter = 0


