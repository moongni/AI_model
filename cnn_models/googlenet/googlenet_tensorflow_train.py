from __future__ import absolute_import, division, print_function, unicode_literals
import os

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt

from googlenet_tensorflow import InceptionV3


tfds.disable_progress_bar()

keras = tf.keras

# Download CIFAR10 DATA
(raw_train, raw_validation, raw_test), metadata = tfds.load(
    'cifar10',
    split=['train[:80%]', 'train[80%:90%]', 'train[90%:]'],
    with_info=True,
    as_supervised=True,
)

print("Train data:", raw_train)
print("Val data:", raw_validation)
print("Test data:", raw_test)

# Image visualize
# get_label_name = metadata.features['label'].int2str

# for image, label in raw_train.take(2).cache().repeat():
#     plt.figure()
#     plt.imshow(image)
#     plt.title(get_label_name(label))

# tf.image 모듈을 사용해 image 포맷
IMG_SIZE = 224
BATCH_SIZE = 32
SHUFFLE_BUFFER_SIZE = 1000


def format_example(image, label):
    image = tf.cast(image, tf.float32)
    image = (image / 127.5) - 1
    image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
    return image, label


train = raw_train.map(format_example)
validation = raw_validation.map(format_example)
test = raw_test.map(format_example)

train_batches = train.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)
validation_batches = validation.batch(BATCH_SIZE)
test_batches = test.batch(BATCH_SIZE)

for image_batch, label_batch in train_batches.take(1):
    break

print("DATA SHAPE IN TEH BATCH:", image_batch.shape)


# Load Model
IMG_SHAPE = (IMG_SIZE, IMG_SIZE, 3)
base_model = InceptionV3(
    include_top=False,
    input_shape=IMG_SHAPE
)

feature_batch = base_model(image_batch)
print(feature_batch.shape)
# base_model.summary()

# FC layer
global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
feature_batch_average = global_average_layer(feature_batch)
print(feature_batch_average.shape)

model = tf.keras.Sequential([
    base_model,
    global_average_layer,
    keras.layers.Dense(1028, activation='relu'),
    keras.layers.Dense(512, activation='relu'),
    keras.layers.Dense(256, activation='relu'),
    keras.layers.Dense(10, activation='softmax'),
])

lr = 0.0001
model.compile(
    optimizer=tf.keras.optimizers.Adam(lr=lr),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
model.summary()


# 모델 훈련
initial_epochs = 10
validation_step = 20

history = model.fit(train_batches,
                    epochs=initial_epochs,
                    validation_data=validation_batches)


# 모델 예측
loss, acc = model.evaluate(test_batches, batch_size=64)
print('loss from test data', loss)
print('accuracy from test data ', acc)