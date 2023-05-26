from __future__ import absolute_import, division, print_function, unicode_literals
import os

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt

from resnet_tensorflow import ResNet50


tfds.disable_progress_bar()

keras = tf.keras

# Download CIFAR10 DATA
(raw_train, raw_validation, raw_test), metadata = tfds.load(
    'cifar10',
    split=['train[:80%]', 'train[80%:90%]', 'train[90%:]'],
    with_info=True,
    as_supervised=True,
)

# (이미지, 레이블) 쌍으로 이루어져 있고 이미지는 3개 채널로 구성되며 레이블은 스칼라로 구성되어 있음
print("Train Data:", raw_train)
print("Val Data:", raw_validation)
print("Test Data:", raw_test)


IMG_SIZE = 224 # 모든 이미지는 224x224으로 크기가 조정됩니다
BATCH_SIZE = 32
SHUFFLE_BUFFER_SIZE = 1000

def format_example(image, label):
  image = tf.cast(image, tf.float32)
  image = (image/127.5) - 1
  image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
  return image, label

train = raw_train.map(format_example)
validation = raw_validation.map(format_example)
test = raw_test.map(format_example)

train_batches = train.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)
validation_batches = validation.batch(BATCH_SIZE)
test_batches = test.batch(BATCH_SIZE)

for image_batch, label_batch in train_batches.take(1):
   pass

print("DATA SHAPE IN ONE BATCH",image_batch.shape)

# Load Model
IMG_SHAPE = (IMG_SIZE, IMG_SIZE, 3)

base_model = ResNet50(input_shape=IMG_SHAPE,
                      include_top=False,
                      weights='imagenet')

global_average_layer = tf.keras.layers.GlobalAveragePooling2D()

# Full Model
model = tf.keras.Sequential([
    base_model,
    global_average_layer,
    keras.layers.Dense(1028, activation='relu'),
    keras.layers.Dense(512, activation='relu'),
    keras.layers.Dense(256, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])

learning_rate = 0.0001
model.compile(optimizer=tf.keras.optimizers.Adam(lr=learning_rate),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

if __name__ == "__main__":
    initial_epochs = 10
    validation_steps=20

    history = model.fit(train_batches,
                        epochs=initial_epochs,
                        validation_data=validation_batches)

    loss, acc = model.evaluate(test_batches, batch_size=64)
    print('loss from test data', loss)
    print('accuracy from test data ', acc)