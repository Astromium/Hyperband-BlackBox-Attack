import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from mlc.datasets.dataset_factory import get_dataset
from keras.layers import Input, Dropout, Dense, Activation
from sklearn.metrics import roc_auc_score, accuracy_score
import joblib

print(tf.__version__)

def create_dnn(units, input_dims):
    model = Sequential()
    model.add(Input(shape=(50)))
    model.add(Dense(units=units[0], activation='relu', input_dim=(50)))
    model.add(Dropout(0.1))
    model.add(Dense(units=units[1], activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(units=units[2], activation='relu'))
    model.add(Dense(units=2))
    model.add(Activation('softmax'))
    
    #model.compile(optimizer=optimizer, loss=loss, metrics=[tf.keras.metrics.BinaryAccuracy()])
    return model

loss_fn = tf.keras.losses.BinaryCrossentropy()
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

LAYERS = [512, 256, 128]
INPUT_DIMS = [64, 1, 756]

model = create_dnn(units=LAYERS, input_dims=INPUT_DIMS)
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
x_test = scaler.transform(x_test)

from tensorflow.keras.utils import to_categorical

train_dataset = tf.data.Dataset.from_tensor_slices((x_train, to_categorical(y_train)))
train_dataset = train_dataset.shuffle(buffer_size=len(x_train)).batch(batch_size=32)
val_dataset = tf.data.Dataset.from_tensor_slices((x_val, to_categorical(y_val)))
val_dataset = val_dataset.batch(batch_size=32)

train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.BinaryAccuracy(name='train_accuracy')
val_loss = tf.keras.metrics.Mean(name='val_loss')
val_accuracy = tf.keras.metrics.BinaryAccuracy(name='val_accuracy')

@tf.function
def train_step(inputs, labels):
    with tf.GradientTape() as tape:
        # Perform forward pass
        logits = model(inputs, training=True)
        # Compute loss
        loss_value = loss_fn(labels, logits)
    
    # Compute gradients
    gradients = tape.gradient(loss_value, model.trainable_variables)
    
    # Update model parameters
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    
    # Update training metrics
    train_loss(loss_value)
    train_accuracy(labels, logits)


@tf.function
def val_step(inputs, labels):
    # Perform forward pass
    logits = model(inputs, training=False)
    # Compute loss
    loss_value = loss_fn(labels, logits)
    
    # Update validation metrics
    val_loss(loss_value)
    val_accuracy(labels, logits)

# Training loop
epochs = 50

for epoch in range(epochs):
    # Reset the metrics
    train_loss.reset_states()
    train_accuracy.reset_states()
    val_loss.reset_states()
    val_accuracy.reset_states()
    
    # Training
    for inputs, labels in train_dataset:
        train_step(inputs, labels)
    
    # Validation
    for inputs, labels in val_dataset:
        val_step(inputs, labels)
    
    print(f'Epoch {epoch+1}/{epochs}:')
    print(f'Training Loss: {train_loss.result():.4f}, Training Accuracy: {train_accuracy.result():.4f}')
    print(f'Validation Loss: {val_loss.result():.4f}, Validation Accuracy: {val_accuracy.result():.4f}')

preds = model.predict(x_test)
classes = np.argmax(preds, axis=1)

roc = roc_auc_score(y_test, classes)
print(f'roc {roc}')
acc = accuracy_score(y_test, classes)
print(f'acc {acc}')

charged_off = np.where(y_test == 1)[0]
print(charged_off.size)

preds = model.predict(x_test[charged_off])
classes = np.argmax(preds, axis=1)
acc = accuracy_score(y_test[charged_off], classes)
print(f'acc charged_off {acc}')