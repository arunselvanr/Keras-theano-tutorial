import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score

#Generate data, Images, 10 classes.
train_no = 100
x_train = np.random.random((train_no, 3, 32, 32))
#y_train = keras.utils.to_categorical(np.random.randint(size=(train_no, 1), low=0, high=10), num_classes=10)
y_train = np.random.randint(size=(train_no, 1), low=0, high=2)

def create_model():
    NN = Sequential()
    NN.add(Conv2D(300, (3,3), input_shape=(3, 32, 32), data_format='channels_first', activation='relu'))
   # NN.add(Conv2D(300, (3,3), activation='relu'))
    NN.add(MaxPool2D(pool_size=(2,2)))
    NN.add(Flatten())
    NN.add(Dropout(0.5))
    NN.add(Dense(1, activation='sigmoid'))
    NN.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return NN

####Time to evaluate
model = KerasClassifier(build_fn=create_model, epochs=10, batch_size=10)
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=123)
result = cross_val_score(model, x_train, y_train, cv=kfold)
print result