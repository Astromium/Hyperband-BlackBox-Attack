import joblib


import pandas as pd
from keras.layers import Input, Dense, Dropout, Activation
from keras.models import Sequential
import tensorflow as tf
import numpy as np

#tf.compat.v1.disable_eager_execution()

scaler = preprocessing_pipeline = joblib.load('./ressources/baseline_scaler.joblib')


def create_dnn(units, optimizer, loss):
    model = Sequential()
    model.add(Input(shape=(63)))
    model.add(Dense(units=units[0], activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(units=units[1], activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(units=units[2], activation='relu'))
    model.add(Dense(units=2))
    model.add(Activation('softmax'))
    
    model.compile(optimizer=optimizer, loss=loss, metrics=[tf.keras.metrics.BinaryAccuracy()])
    return model

opt = tf.keras.optimizers.Adam()
loss = tf.keras.losses.CategoricalCrossentropy()

LAYERS = [256, 128, 64]
model = create_dnn(units=LAYERS, optimizer='adam', loss=loss)
print(model.summary())




df = pd.read_csv('./ressources/url.csv')
y = np.array(df['is_phishing'])
X = np.array(df[df.columns.drop('is_phishing')])

from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

merged = np.arange(len(X))
i_train, i_test = train_test_split(
        merged,
        random_state=100,
        shuffle=True,
        stratify=y[merged],
        test_size=0.2,
)
i_train, i_val = train_test_split(
        i_train,
        random_state=200,
        shuffle=True,
        stratify=y[i_train],
        test_size=0.2,
)

X = scaler.fit_transform(X)
X_train, X_test, X_val = X[i_train], X[i_test], X[i_val]
y_train, y_test, y_val = y[i_train], y[i_test], y[i_val]
y_train_cat, y_val_cat = to_categorical(y_train), to_categorical(y_val)
y_test_cat = to_categorical(y_test)


history = model.fit(X_train, y_train_cat, epochs=30, validation_data=(X_val, y_val_cat))
save_options = tf.saved_model.SaveOptions(experimental_io_device='/job:localhost')
model.save('model_url2.h5', options=save_options)
#joblib.dump(model, './ressources/model_url.pkl')


#model = joblib.load('./ressources/model_url.pkl')