import time

from common import *

print 'Check training with sparse data'
# get sparse data
start_time = time.time()
X = get_x_data_sparse()
print 'Reading data time', time.time()-start_time
y = get_y_data()
X_train, Y_train, X_test, Y_test = split_data(X, y)

# construct network
model = get_model()

# start training
start_time = time.time()
model.fit_generator(generator=sparse_generator(X_train, Y_train, batch_size, True), 
                    samples_per_epoch = len(X_train), nb_epoch = nb_epoch, verbose=1,
		    validation_data=sparse_generator(X_test, Y_test, batch_size, False),
		    nb_val_samples=len(X_test), max_q_size=20, nb_worker=12)
print 'Training time', time.time()-start_time
# evaluate network
score = model.evaluate_generator(sparse_generator(X_test, Y_test, batch_size, False), 
		val_samples=len(X_test))
print('Test logloss:', score)
