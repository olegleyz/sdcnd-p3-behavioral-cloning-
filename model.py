from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten, Dropout, Lambda, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.wrappers.scikit_learn import KerasRegressor
from keras.callbacks import EarlyStopping
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# load data
label_file = '../data/p3/data/driving_log.csv'
data_dir = '../data/p3/data/'
data = pd.read_csv(label_file)[['center','left','right','steering']]

# prepare dataset
arr_size = 6*len(data[data.steering!=0])+len(data[data.steering==0])
X_train=np.empty((arr_size,66,220,3))
y_train=np.empty((arr_size))

koef=1.1

for i,val in enumerate(data[data.steering!=0].iterrows()):        
        X_train[6*i] = plt.imread(data_dir + val[1].center.strip())[74:74+66,50:270,:]
        y_train[6*i] = val[1].steering

        X_train[6*i+1] = X_train[2*i][:,::-1]
        y_train[6*i+1] = -1*y_train[2*i]

        if val[1].steering>0:
            X_train[6*i+2] = plt.imread(data_dir + val[1].left.strip())[74:74+66,50:270,:]
            y_train[6*i+2] = val[1].steering*koef

            X_train[6*i+3] = plt.imread(data_dir + val[1].right.strip())[74:74+66,50:270,:]
            y_train[6*i+3] = val[1].steering/koef

            X_train[6*i+4] = X_train[6*i+2][:,::-1]
            y_train[6*i+4] = y_train[6*i+2]*-1

            X_train[6*i+5] = X_train[6*i+3][:,::-1]
            y_train[6*i+5] = y_train[6*i+3]*-1
        else:
            X_train[6*i+2] = plt.imread(data_dir + val[1].left.strip())[74:74+66,50:270,:]
            y_train[6*i+2] = val[1].steering/koef

            X_train[6*i+3] = plt.imread(data_dir + val[1].right.strip())[74:74+66,50:270,:]
            y_train[6*i+3] = val[1].steering*koef

            X_train[6*i+4] = X_train[6*i+2][:,::-1]
            y_train[6*i+4] = y_train[6*i+2]*-1

            X_train[6*i+5] = X_train[6*i+3][:,::-1]
            y_train[6*i+5] = y_train[6*i+3]*-1
        
        
for i,val in enumerate(data[data.steering==0].iterrows()):
        X_train[i + 6*len(data[data.steering!=0])] = plt.imread(data_dir + val[1].center.strip())[74:74+66,50:270,:]
        y_train[i + 6*len(data[data.steering!=0])] = 0.0


# train deep neural network
akt = 'relu'
init = 'he_normal'

model = Sequential()
model.add(Lambda(lambda x: x/128 - 1., input_shape=(66, 220, 3), output_shape=(66, 220, 3)))
model.add(Convolution2D(24, 5, 5, subsample = (2, 2), input_shape=(66, 220, 3),
                        border_mode='valid',activation=akt, init=init))
model.add(Convolution2D(36, 5, 5, subsample = (2, 2), border_mode='valid',activation=akt, init=init))
model.add(Convolution2D(48, 5, 5, subsample = (2, 2), border_mode='valid',activation=akt, init=init))

model.add(Convolution2D(64, 3, 3, subsample = (1, 1), border_mode='valid',activation=akt, init=init))
model.add(Convolution2D(64, 3, 3, subsample = (1, 1), border_mode='valid',activation=akt, init=init))

model.add(Flatten())
model.add(Dense(100,activation=akt, init=init))
model.add(Dense(50, activation=akt, init=init))
model.add(Dense(10, activation=akt, init=init))
model.add(Dense(1, init=init))

model.compile(loss='mean_squared_error', optimizer='adam')


from keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='val_loss', patience=2)

model.fit(np.asarray(X_train), np.asarray(y_train), nb_epoch=10, batch_size=32, validation_split=0.2,
          callbacks=[early_stopping])


# save the model and weights
json_string = model.to_json()
model.save_weights("model.h5")

with open("model.json", "w") as json_file:
    json_file.write(json_string)
print("Saved model to disk")