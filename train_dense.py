import time

from common import *

print 'Check training with dense data'
# get sparse data
start_time = time.time()
X = get_x_data_dense()
print 'Reading data time', time.time()-start_time
y = get_y_data()
X_train, Y_train, X_test, Y_test = split_data(X, y)

# construct network
model = get_model()

# start training
start_time = time.time()
model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch, 
	  verbose=1, validation_data=(X_test, Y_test), shuffle=True)
print 'Training time', time.time()-start_time
# evaluate network
score = model.evaluate(X_test, Y_test)
print('Test logloss:', score)
