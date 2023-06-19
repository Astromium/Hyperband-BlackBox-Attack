import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from mlc.datasets.dataset_factory import get_dataset
from keras.layers import Input, Dropout, Dense, Activation

print(tf.__version__)

def create_dnn(units, input_dims, optimizer, loss):
    model = Sequential()
    model.add(Input(shape=(50)))
    model.add(Dense(units=units[0], activation='relu', input_dim=(50)))
    model.add(Dropout(0.1))
    model.add(Dense(units=units[1], activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(units=units[2], activation='relu'))
    model.add(Dense(units=2))
    model.add(Activation('softmax'))
    
    model.compile(optimizer=optimizer, loss=loss, metrics=[tf.keras.metrics.BinaryAccuracy()])
    return model

opt = keras.optimizers.Adam()
loss = keras.losses.CategoricalCrossentropy()

LAYERS = [128, 64, 32]
INPUT_DIMS = [64, 1, 756]

model = create_dnn(units=LAYERS, input_dims=INPUT_DIMS, optimizer=opt, loss=loss)
print(model.summary())

ds = get_dataset('lcld_v2_time')
splits = ds.get_splits()

x, y = ds.get_x_y()

categorical = ['home_ownership', 'verification_status', 'purpose', 'initial_list_status', 'application_type']
encoded_df = pd.get_dummies(x, columns=categorical)
encoded_df = encoded_df.to_numpy()
x_train, y_train = encoded_df[splits['train']], y[splits['train']]
x_val, y_val = encoded_df[splits['val']], y[splits['val']]
x_test, y_test = encoded_df[splits['test']], y[splits['test']]

print(f'train dataset {x_train.shape}')
print(f'test dataset {x_test.shape}')
print(f'validation dataset {x_val.shape}')

from tensorflow.keras.utils import to_categorical
scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_val = scaler.transform(x_val)

history = model.fit(x_train, to_categorical(y_train), epochs=10, validation_data=(x_val, to_categorical(y_val)))

x_test = scaler.transform(x_test)
evaluation = model.evaluate(x_test, to_categorical(y_test))
print(f'Evaluation : {evaluation}')

from sklearn.metrics import roc_auc_score, accuracy_score

preds = model.predict(x_test)
classes = np.argmax(preds, axis=1)

roc = roc_auc_score(y_test, classes)
print(f'roc {roc}')
acc = accuracy_score(y_test, classes)
print(f'acc {acc}')

model.save('./ressources/custom_lcld_model.h5')