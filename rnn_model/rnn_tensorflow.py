import os
import time

import numpy as np
import tensorflow as tf


path_to_file = tf.keras.utils.get_file('shakespeare.txt', 'https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt')

# 파이썬 2와 호환될 수 있도록 디코딩
text = open(path_to_file, 'rb').read().decode(encoding='utf-8')

# 고유 문자 수
vocab = sorted(set(text))

# text to vector
char2idx = {u: i for i, u in enumerate(vocab)}
idx2char = np.array(vocab)

text_as_int = np.array([char2idx[c] for c in text])

# split train & train target
seq_length = 100
examples_per_epoch = len(text) // seq_length

# 훈련 샘플/타겟 만들기
char_dataset = tf.data.Dataset.from_tensor_slices(text_as_int)

sequences = char_dataset.batch(seq_length+1, drop_remainder=True)


# map method를 통해 각 배치에서 입력 텍스트 마지막 요소 제거, 타겟 첫번째 요소 제거
def split_input_target(chunk):
    input_text = chunk[:-1]
    target_text = chunk[1:]
    return input_text, target_text


dataset = sequences.map(split_input_target)

# 훈련
BATCH_SIZE = 64

# 데이터셋을 섞을 버퍼
# (TF 데이터는 무한한 시퀀스와 함께 작동하도록 설계됨)
# 따라서 전체 시퀀스를 메모리에 섞지 않음 대신 요소 섞는 버퍼 유지
BUFFER_SIZE = 10000

dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)

## 모델 설계
## - `tf.keras.layers.Embedding` : 입력층. `embedding_dim` 차원 벡터에 각 문자의 정수 코드를 매핑하는 훈련 가능한 검색 테이블.
## - `tf.keras.layers.LSTM` : 크기가 `units = rnn_units`인 RNN의 유형
## - `tf.keras.layers.Dense` : 크기가 `vocab_size`인 출력을 생성하는 출력층. 

# 문자로 된 어휘 사전 크기
vocab_size = len(vocab)

# 임베딩 차원
embedding_dim = 256

# RNN 유닛 개수
rnn_units = 1024


def build_model(vocab_size, embedding_dim, rnn_units, batch_size):
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, embedding_dim,
                                  batch_input_shape=[batch_size, None]),
        tf.keras.layers.LSTM(rnn_units,
                             return_sequences=True,
                             stateful=True,
                             recurrent_initializer='glorot_uniform'),
        tf.keras.layers.Dense(vocab_size)
    ])
    return model


model = build_model(
    vocab_size=len(vocab),
    embedding_dim=embedding_dim,
    rnn_units=rnn_units,
    batch_size=BATCH_SIZE
)
model.summary()


def loss(labels, logits):
    return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)

model.compile(optimizer='adam', loss=loss)
# 훈련 체크포인트
checkpoint_dir = './rnn_model/training_checkpoint'
# 체크포인트 파일 이름
checkpoint_prefix = os.path.join(checkpoint_dir, 'ckpt_{epoch}')

checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_prefix,
    save_weights_only=True
)

EPOCHS = 10
history = model.fit(dataset, epochs=EPOCHS, callbacks=[checkpoint_callback])

model = build_model(vocab_size, embedding_dim, rnn_units, batch_size=1)
model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))
model.build(tf.TensorShape([1, None]))
model.summary()


# 평가 모델사용하여 텍스트 생성
def generate_text(model, start_string):
    # 생성할 문자 수
    num_generate = 1000

    # 시작 문자열 벡터화
    input_eval = [char2idx[s] for s in start_string]
    input_eval = tf.expand_dims(input_eval, 0)

    # 결과 저장 빈 문자열
    text_generated = []

    # 온도가 낮으면 더 예측 가능한 텍스트가 됩니다.
    # 온도가 높으면 더 의외의 텍스트가 됩니다.
    # 최적 세팅을 찾기 위한 실험
    temperature = 1.0

    # 배치 크기 1
    model.reset_states()
    for i in range(num_generate):
        predictions = model(input_eval)
        # 배치 차원 제거
        predictions = tf.squeeze(predictions, 0)

        # 범주형 분포를 사용하여 모델에서 리턴한 단어 예측
        predictions = predictions / temperature
        predicted_id = tf.random.categorical(predictions, num_samples=1)[-1, 0].numpy()

        # 예측된 단어를 다음 입력으로 모델에 전달
        input_eval = tf.expand_dims([predicted_id], 0)
        text_generated.append(idx2char[predicted_id])

    return (start_string + ''.join(text_generated))


print(generate_text(model, start_string=u"ROMEO: "))


if __name__ == "__main__":
    # print(tf.__version__)

    # # RNN Test
    # inputs = np.random.random([32, 10, 8]).astype(np.float32)

    # rnn = tf.keras.layers.RNN(
    #     tf.keras.layers.SimpleRNNCell(4),
    #     return_sequences=True,
    #     return_state=True
    # )

    # # whole sequence output has shape [32, 10, 4]
    # # final state has shape [32, 4]
    # whole_sequence_output, final_state = rnn(inputs)
    # print(whole_sequence_output.shape)
    # print(final_state.shape)

    # # LSTM Test
    # inputs = tf.random.normal([32, 10, 8])
    # rnn = tf.keras.layers.RNN(
    #     tf.keras.layers.LSTMCell(4),
    #     return_sequences=True,
    #     return_state=True
    # )
    # whole_seq_output, final_memory_state, final_carry_state = rnn(inputs)

    # print(whole_seq_output.shape)
    # print(final_memory_state.shape)
    # print(final_carry_state.shape)

    # 텍스트 길이
    print(f"텍스트의 길이: {len(text)}자")
    # 처음 250자
    print(text[:250])
    # 파일 고유 문자 수
    print(f"고유 문자 수 {len(vocab)}자")
    # 각 문자에 대한 정수 표현을 만듦
    print('{')
    for char,_ in zip(char2idx, range(20)):
        print('  {:4s}: {:3d},'.format(repr(char), char2idx[char]))
    print('  ...\n}')
    # 텍스트에서 처음 13개의 문자가 숫자로 어떻게 매핑되었는지를 보여줍니다
    print ('{} ---- 문자들이 다음의 정수로 매핑되었습니다 ---- > {}'.format(repr(text[:13]), text_as_int[:13]))
    # 훈련 셋 5개 확인
    for i in char_dataset.take(5):
        print(idx2char[i.numpy()])
    # 배치크기 시퀀스 확인
    for item in sequences.take(5):
        print(repr(''.join(idx2char[item.numpy()])))
        print(len(item.numpy()))
    # # 전처리 된 텍스트 확인
    # for input_example, target_example in dataset.take(1):
    #     print ('입력 데이터: ', repr(''.join(idx2char[input_example.numpy()])))
    #     print ('타깃 데이터: ', repr(''.join(idx2char[target_example.numpy()])))
    # for i, (input_idx, target_idx) in enumerate(zip(input_example[:5], target_example[:5])):
    #     print("{:4d}단계".format(i))
    #     print("  입력: {} ({:s})".format(input_idx, repr(idx2char[input_idx])))
    #     print("  예상 출력: {} ({:s})".format(target_idx, repr(idx2char[target_idx])))
    # 모델 output 확인
    for input_example_batch, target_example_batch in dataset.take(1):
        example_batch_predictions = model(input_example_batch)
        print(example_batch_predictions.shape, "# (배치 크기, 시퀀스 길이, 어휘 사전 크기)")
        # 배치의 첫 번째 샘플링
        sampled_indices = tf.random.categorical(example_batch_predictions[0], num_samples=1)
        sampled_indices = tf.squeeze(sampled_indices,axis=-1).numpy()
        print(sampled_indices)
        # 훈련되지 않은 모델에 의해 예측된 텍스트
        print("입력: \n", repr("".join(idx2char[input_example_batch[0]])))
        print()
        print("예측된 다음 문자: \n", repr("".join(idx2char[sampled_indices])))
        example_batch_loss = loss(target_example_batch, example_batch_predictions)
        print("예측 배열 크기(shape): ", example_batch_predictions.shape, " # (배치 크기, 시퀀스 길이, 어휘 사전 크기")
        print("스칼라 손실:          ", example_batch_loss.numpy().mean())