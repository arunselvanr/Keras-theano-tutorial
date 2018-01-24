import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Dropout, Dense, Flatten
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score #pip install -U scikit-learn

########Let's generate some random data, simple sequences with just two classes.
train_sample =100
#Since we are going to use cross_validation of keras we don't need any test data.
#The idea is that we have very little data and we need to learn from it.
#The robustness of our model can be learned using cross_validation after creating
#a certain number of folds.
x_train = np.random.random((train_sample, 1))
y_train = np.random.randint(size=(train_sample, 1), low=0, high=2)

def create_model():
    model = Sequential()
    model.add(Dense(128, input_shape=(1,), activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

model = KerasClassifier(build_fn=create_model, epochs=100, batch_size=10)
#KerasClassifier is when the NN performs the function of a classifier.
#KerasRegressor is when the NN performs the funciton of a regressor.

kfold = StratifiedKFold(n_splits=30, shuffle=True, random_state=12)
#StratifiedKFold can be found in sklearn.model_selection. It's job is to split
#the little data available into train and test data. n_splits creates the said
#number of folds. Shuffle is a boolean variable that decides if the data is shuffled or not.
#random_state takes int or None. This is just a seed.
result = cross_val_score(model, x_train, y_train, cv=kfold)

print result