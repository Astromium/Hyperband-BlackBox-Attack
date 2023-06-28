import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from mlc.datasets.dataset_factory import get_dataset
from keras.layers import Input, Dropout, Dense, Activation
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline

def create_dnn(units, input_dims, optimizer, loss):
    model = Sequential()
    model.add(Input(shape=(50)))
    model.add(Dense(units=units[0], activation='relu'))
    model.add(Dense(units=units[1], activation='relu'))
    model.add(Dense(units=units[2], activation='relu'))
    model.add(Dense(units=2, activation='softmax'))
    #model.add(Activation('softmax'))
    
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy', tf.keras.metrics.Recall(), tf.keras.metrics.AUC()])
    return model

opt = keras.optimizers.SGD()
loss = keras.losses.CategoricalCrossentropy()

LAYERS = [64, 32, 16]
INPUT_DIMS = [64, 1, 756]

model = create_dnn(units=LAYERS, input_dims=INPUT_DIMS, optimizer=opt, loss=loss)
print(model.summary())

ds = get_dataset('lcld_v2_iid')
splits = ds.get_splits()

x, y = ds.get_x_y()

categorical = ['home_ownership', 'verification_status', 'purpose', 'initial_list_status', 'application_type']

#x[categorical] = x[categorical].astype(str)

num_indices = [x.columns.get_loc(col) for col in numerical]
col_order = list(numerical) + list(categorical)
x = x[col_order]
cat_indices = [x.columns.get_loc(col) for col in categorical]
numerical = [col for col in x.columns if col not in categorical]
print(f'cat indices {cat_indices}')
#encoded_df = pd.get_dummies(x, columns=categorical)
#encoded_df = encoded_df.to_numpy()
x_train, y_train = x.iloc[splits['train']], y[splits['train']]
x_test, y_test = x.iloc[splits['test']], y[splits['test']]
x_val, y_val = x.iloc[splits['val']], y[splits['val']]

#x_train, y_train = encoded_df[splits['train']], y[splits['train']]

#x_val, y_val = encoded_df[splits['val']], y[splits['val']]

#x_test, y_test = encoded_df[splits['test']], y[splits['test']]

print(f'train dataset {x_train.shape}')
print(f'test dataset {x_test.shape}')
print(f'validation dataset {x_val.shape}')

num_transformer = MinMaxScaler()
cat_transformer = OneHotEncoder(sparse=False)

preprocessor = ColumnTransformer(
    transformers=[
        ('num', num_transformer, num_indices),
        ('cat', cat_transformer, cat_indices)
    ]
)

preprocessor.fit(x)
joblib.dump(preprocessor, './ressources/lcld_preprocessor.joblib')

x_train = preprocessor.transform(x_train)
print(x_train.shape)
x_val = preprocessor.transform(x_val)
print(x_val.shape)
x_test = preprocessor.transform(x_test)
print(x_test.shape)



from tensorflow.keras.utils import to_categorical

history = model.fit(x_train, to_categorical(y_train), epochs=10, validation_data=(x_val, to_categorical(y_val)))

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
#joblib.dump(scaler, './ressources/custom_lcld_scaler.joblib')
