import numpy as np
import theano, keras
from theano import tensor as T
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import  SGD
import os


train_sample = 1000
test_sample = 100
#np.random.random will return an output size (1000, 20) of uniformly distributed independent random variables.
#np.random.randint(size, low, high) will output a size (1000, 20) uniformly distributed, over intergers,
#random variables that range from low to high-1. Note that high is not included
x_train = np.random.random(size=(train_sample, 20))
#keras.utils.to_categorical(vector of size nx1, num_classes) converts a column vector into a category matrix
#with column size equalling num_classes.
#For example if num_classes = 2, then each entry in the column vector is 0 or 1. Suppose an entry is 1,
#in the category model it is now 01. If it were 0, it corresponds to 10.
y_train = keras.utils.to_categorical(np.random.randint(size=(train_sample, 1), low=0, high=10), num_classes=10)
x_test =  np.random.random((test_sample, 20))
y_test = keras.utils.to_categorical(np.random.randint(size=(test_sample, 1), low=0, high=10))


#Now we are ready to build our model for multi-category classification.
#Let's go for two hidden layers but we allow for regularization through dropouts.
#Our deep neural network will be input--->hidden_layer_1--->(with dropout)hidden_layer_1--->(with_dropout)output_layer(soft_max)
NN = Sequential()
NN.add(Dense(128, input_shape=(20,), activation='relu'))#Here the input shape is a simple vector (list). Hence (20, ).
#Note that we may equivalently write input_dim = 20
NN.add(Dropout(.4))
NN.add(Dense(128, activation='relu'))
NN.add(Dropout(.4))
NN.add(Dense(10, activation='softmax'))
#SGD is a keras optimizer that needs to some prepping. lr is the learning_rate, decay is the decay_rate
#of the learning rate, momentum term and if we are to use nesterov optimization.
sgd = SGD(lr = .01, decay=1e-6, momentum=.9, nesterov=True)
#Other optimizers: 'rmsprop', 'adam', etc.
#Other loss functions: 'binary_crossentropy', 'mse' (mean squared error), etc.
NN.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])

#Now we are ready to fit the training data. Since the data size is typically large, it is not wise to
#consider all of them at once. Let us say the loss function is the average mean squared error. Then we can only make one
#gradient step. If the batch_size were 50 and epochs were 10, the number of gradient descent steps would be 200.
#However each step would involve a loss function that is the average of 50 sample points. Averaging typically
#reduces biases in training the neural network at hand.
NN.fit(x_train, y_train, batch_size=128, epochs=1)

#What is left to do is evaluate our neural network on the test data.

score = NN.evaluate(x_test, y_test, batch_size=128)
print score
#os.system('spd-say "Your program is done"')

##################################################################################################################
##################################################################################################################
#################################The Binary Classification Problem#############################################
##################################################################################################################
##################################################################################################################

x_train1 = np.random.random((train_sample, 20))
y_train1 = np.random.randint(size=(train_sample, 1), low=0, high=2)
x_test1 = np.random.random((test_sample, 20))
y_test1 = np.random.randint(size=(test_sample, 1), low=0, high=2)
#We intend to utilize binary_crossentropy as our loss function. Here the output layer is a single sigmoid neuron.
#Hence we do not need to_categorical it.


NN1 = Sequential()
NN1.add(Dense(128, input_shape=(20,), activation='relu'))
NN1.add(Dense(1, activation='sigmoid'))
NN1.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

NN1.fit(x_train1, y_train1, batch_size=256, epochs=1) #Please tinker with the epochs and batch_size.
score1 =  NN1.evaluate(x_test1, y_test1, batch_size=256) #The epochs definitely needs to be much much higher.
print score1

##################################################################################################################
##################################################################################################################
####################################Convolutional neural network##################################################
##################################################################################################################
##################################################################################################################
from keras.layers import Conv2D, MaxPool2D, Flatten
#When processing images using theano as the backend, it is worth noting that data_format = 'channels_first' needs
#to be specified. In other words it expects (channels, height, width) not channels last. Else, we need to reshape data.
#Example: x_train = x_train.reshape(x_train[0], channels, height, width).

#Dummy data dummy data dummy data for training and testing.
x_train2 = np.random.random((train_sample, 3, 32, 32))
y_train2 = keras.utils.to_categorical(np.random.randint(size=(train_sample, 1), low=0, high=10), num_classes=10)
x_test2 = np.random.random((test_sample, 3, 32, 32))
y_test2 = keras.utils.to_categorical(np.random.randint(size=(test_sample, 1), low= 0, high=10), num_classes=10)


NN2 = Sequential()
NN2.add(Conv2D(64, (3,3), activation='relu', input_shape=x_train2.shape[1:], data_format='channels_first'))
#If tensorflow were to be used as a backend, then we would use the 'channels_first' data_format.
NN2.add(Conv2D(64, (3,3), activation='relu'))
NN2.add(MaxPool2D(pool_size=(2,2)))
NN2.add(Flatten())
NN2.add(Dropout(0.5))
NN2.add(Dense(128, activation='relu'))
NN2.add(Dense(10, activation='softmax'))
NN2.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

NN2.fit(x_train2, y_train2, batch_size=32, epochs=10)
score2 = NN2.evaluate(x_test2, y_test2, batch_size=32)
print score2