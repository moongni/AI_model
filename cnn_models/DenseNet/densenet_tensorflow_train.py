from __future__ import absolute_import, division, print_function, unicode_literals
import os

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from densenet_tensorflow import DenseNet

keras = tf.keras
print("tensorflow version",tf.__version__)

IMG_SIZE = 224 # 모든 이미지는 224x224으로 크기 조정 
EPOCHS = 2
BATCH_SIZE=16
learning_rate = 0.0001

from keras.datasets import cifar10
from keras.utils import np_utils
import tensorflow_datasets as tfds

tfds.disable_progress_bar()

#분류할 클래스 개수 
num_classes=10 # Cifar10의 클래스 개수

(raw_train, raw_validation, raw_test), metadata = tfds.load(
    'cifar10',
    split=['train[:90%]', 'train[90%:]', 'test'],
    with_info=True,
    as_supervised=True,
)

print("Train data 개수:",len(raw_train))
print("Val data 개수:",len(raw_validation))
print("Test data 개수:",len(raw_test))


def format_example(image, label):
  image = tf.cast(image, tf.float32)
  image = (image/127.5) - 1
  image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
  return image, label


# #map 함수를 사용하여 데이터셋의 각 항목에 데이터 포맷 함수를 적용
train = raw_train.map(format_example)
validation = raw_validation.map(format_example)
test = raw_test.map(format_example)

SHUFFLE_BUFFER_SIZE = 1000
train_batches = train.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)
validation_batches = validation.batch(BATCH_SIZE)
test_batches = test.batch(BATCH_SIZE)

IMG_SHAPE = (IMG_SIZE, IMG_SIZE, 3)

#CNN 모델 변경하려면 여기서 변경
#ImageNet으로 사전 훈련된 모델 불러오기
base_model = tf.keras.applications.DenseNet121(input_shape=IMG_SHAPE,
                                               include_top=False,
                                                classes=1000,
                                               weights='imagenet')

#GAP 층
global_average_layer = tf.keras.layers.GlobalAveragePooling2D(name='avg_pool')
#분류 층
prediction_layer=keras.layers.Dense(num_classes, activation='softmax',name='predictions')

model = tf.keras.Sequential([
  base_model,
  global_average_layer,
  prediction_layer
])

model.summary()

model.compile(optimizer=tf.keras.optimizers.Adam(lr=learning_rate),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

if __name__ == "__main__":
    history = model.fit(train_batches,
                        epochs=EPOCHS,
                        validation_data=validation_batches,
                        batch_size=BATCH_SIZE)
    loss_and_metrics = model.evaluate(test_batches, batch_size=64)
    print("테스트 성능 : {}%".format(round(loss_and_metrics[1]*100,4)))