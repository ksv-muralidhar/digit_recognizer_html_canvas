import tensorflow as tf
import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping


(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
np.random.seed(42)
tf.random.set_seed(42)
shuffle_idx = np.arange(len(x_train))
np.random.shuffle(shuffle_idx)
x_train = x_train[shuffle_idx]
y_train = y_train[shuffle_idx]

x_train = x_train.reshape(len(x_train), 28, 28, 1)
x_test = x_test.reshape(len(x_test), 28, 28, 1)

train_datagen = ImageDataGenerator(height_shift_range=0.1,
                                   width_shift_range=0.25,
                                   rotation_range=30,
                                   zoom_range=0.2,
                                   rescale=1. / 255,
                                   fill_mode='nearest')
test_datagen = ImageDataGenerator(rescale=1. / 255)
tf.random.set_seed(11)
cnn = tf.keras.models.Sequential()
cnn.add(tf.keras.layers.Conv2D(filters=128, kernel_size=3, strides=1,
                               data_format='channels_last',
                               activation='relu',
                               input_shape=(28, 28, 1),
                               padding="same"))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2)))
cnn.add(tf.keras.layers.Conv2D(filters=128, kernel_size=3, strides=1,
                               data_format='channels_last',
                               activation='relu', padding="same"))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2)))
cnn.add(tf.keras.layers.Flatten())
cnn.add(tf.keras.layers.Dense(128, activation="relu"))
cnn.add(tf.keras.layers.Dropout(0.2))
cnn.add(tf.keras.layers.Dense(128, activation="relu"))
cnn.add(tf.keras.layers.Dropout(0.2))
cnn.add(tf.keras.layers.Dense(10, activation="softmax"))
cnn.compile(optimizer="adam", loss="sparse_categorical_crossentropy",
            metrics="sparse_categorical_accuracy")
early_stop = EarlyStopping(monitor='val_loss', patience=5)
training_data = train_datagen.flow(x_train, y_train, batch_size=32, seed=42)
validation_data = test_datagen.flow(x_test, y_test, batch_size=32, seed=42, shuffle=False)
history = cnn.fit(training_data, epochs=500, callbacks=[early_stop],
                  validation_data=validation_data)
cnn.save(os.path.join('models', 'mnist.h5'))
