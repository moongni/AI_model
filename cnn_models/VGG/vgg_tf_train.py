from __future__ import absolute_import, division, print_function, unicode_literals
import os
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf


keras = tf.keras

print("tensorflow version", tf.__version__)

IMG_SIZE = 224 # 모든 이미지 224 x 224로 크기 고정
EPOCHS = 3
BATCH_SIZE = 128
learning_rate = 0.0001


# 데이터 세트 다운로드 및 탐색
from keras.datasets import cifar10
from keras.utils import np_utils
import tensorflow_datasets as tfds


tfds.disable_progress_bar()

# 타겟 클래스 수
num_classes = 10

(raw_train, raw_validation, raw_test), metadata = tfds.load(
    'cifar10',
    split=['train[:90%]', 'train[90%:]', 'test'],
    with_info=True,
    as_supervised=True,
)

print("Train data :", len(raw_train))
print("Val data :", len(raw_validation))
print("Test data :", len(raw_test))

# 데이터 정규화 (tf.image 모듈 사용)
def format_example(image, label):
    image = tf.cast(image, tf.float32)
    image = (image / 127.5) - 1
    image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
    return image, label


train = raw_train.map(format_example)
validation = raw_validation.map(format_example)
test = raw_test.map(format_example)

# 데이터 세트 만들기
SHUFFLE_BUFFER_SIZE = 1000
train_batches = train.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)
validation_batches = validation.batch(BATCH_SIZE)
test_batches = test.batch(BATCH_SIZE)
print("All data set makes batches")

# 데이터 가시화
# get_label_name = metadata.features['label'].int2str

# for image, label in raw_train.take(2).cache().repeat():
#     plt.figure()
#     plt.imshow(image)
#     plt.title(get_label_name(label))

if __name__ == "__main__":
    IMG_SHAPE = (IMG_SIZE, IMG_SIZE, 3)
    # ImageNet으로 사전 훈련된 모델 불러오기
    base_model = tf.keras.applications.VGG16(input_shape=IMG_SHAPE,
                                             include_top=True,
                                             classes=1000,
                                             weights='imagenet')
    model = tf.keras.Sequential()
    for layer in base_model.layers[:-1]:
        model.add(layer)
    model.add(keras.layers.Dense(num_classes, activation='softmax', name='predictions'))

    model.summary()

    model.compile(optimizer=tf.keras.optimizers.Adam(lr=learning_rate),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    history = model.fit(train_batches,
                        epochs=EPOCHS,
                        validation_data=validation_batches,
                        batch_size=BATCH_SIZE)
    
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    plt.figure(figsize=(8, 8))
    plt.subplot(2, 1, 1)
    plt.plot(acc, label='Training Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.ylabel('Accuracy')
    plt.ylim([min(plt.ylim()),1])
    plt.title('Training and Validation Accuracy')

    plt.subplot(2, 1, 2)
    plt.plot(loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.ylabel('Cross Entropy')
    plt.ylim([0,1.0])
    plt.title('Training and Validation Loss')
    plt.xlabel('epoch')
    plt.show()

    # TEST SET 테스트
    loss_and_metrics = model.evaluate(test_batches, batch_size=64)
    print("테스트 성능 : {}%".format(round(loss_and_metrics[1]*100,4)))