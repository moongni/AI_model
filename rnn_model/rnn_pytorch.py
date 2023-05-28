# 데이터 링크
# https://download.pytorch.org/tutorial/data.zip
from __future__ import unicode_literals, print_function, division
from io import open
import os
import time
import math
import glob
import string
import random
import unicodedata

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


CUR_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    'data', 'names', '*.txt'
)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def findFiles(path):
    return glob.glob(path)


all_letters = string.ascii_letters + " .,;"
n_letters = len(all_letters)


# 유니코드 문자열을 ASCII로 변환 https://stackoverflow.com/a/518232/2809427
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn' and c in all_letters
    )


# 각 언어의 이름 목록인 category_lines 사전 생성
category_lines = {}
all_categories = []


# 파일을 읽고 줄 단위 분리
def readLines(filename):
    lines = open(filename, encoding='utf-8').read().strip().split('\n')
    return [unicodeToAscii(line) for line in lines]


for filename in findFiles(CUR_PATH):
    category = os.path.splitext(os.path.basename(filename))[0]
    all_categories.append(category)
    lines = readLines(filename)
    category_lines[category] = lines

n_categories = len(all_categories)


# 이름을 Tensor -> <1 x n_letters> 크기의 원핫벡터 생성
def letterToIndex(letter):
    return all_letters.find(letter)


def letterToTensor(letter):
    tensor = torch.zeros(1, n_letters)
    tensor[0][letterToIndex(letter)] = 1
    return tensor


def lineToTensor(line):
    tensor = torch.zeros(len(line), 1, n_letters)
    for li, letter in enumerate(line):
        tensor[li][0][letterToIndex(letter)] = 1
    return tensor


# RNN 모델 작성
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()

        self.hidden_size = hidden_size

        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        combined = torch.cat((input, hidden), 1)
        hidden = self.i2h(combined)
        output = self.i2o(combined)
        output = self.softmax(output)
        return output, hidden
    
    def initHidden(self):
        return torch.zeros(1, self.hidden_size)
    

n_hidden = 128
rnn = RNN(n_letters, n_hidden, n_categories)


# 학습 준비
def categoryFromOutput(output):
    top_n, top_i = output.topk(1) # 텐서의 가장 큰 값 및 주소
    category_i = top_i[0].item()
    return all_categories[category_i], category_i


# 학습 예시
def randomChoice(l):
    return l[random.randint(0, len(l) - 1)]


def randomTrainingExample():
    category = randomChoice(all_categories)
    line = randomChoice(category_lines[category])
    category_tensor = torch.tensor([all_categories.index(category)], dtype=torch.long)
    line_tensor = lineToTensor(line)
    return category, line, category_tensor, line_tensor


"""RNN 마지막 layer가 nn.LogSoftmax이므로 nn.NLLLoss 사용 (nn.CrossEntropyLoss = nn.LogSoftmax + nn.NLLLoss)
1. 입력과 정답 tensor 생성
2. 0으로 초기화된 은닉상태 생성
3. 각 문자 읽기 & 다음 문자를 위한 은닉 상태 유지
4. 목표와 최종 출력 비교
5. 역전파
6. 출력과 손실 반환
"""
criterion = nn.NLLLoss()
criterion.to(DEVICE)
learning_rate = 0.005

rnn = RNN(n_letters, n_hidden, n_categories)
rnn.to(DEVICE)


def train(category_tensor, line_tensor):
    hidden = rnn.initHidden().to(DEVICE)

    rnn.zero_grad()

    for i in range(line_tensor.size()[0]):
        output, hidden = rnn(line_tensor[i], hidden)
    
    loss = criterion(output, category_tensor)
    loss.backward()

    for p in rnn.parameters():
        p.data.add_(p.grad.data, alpha=-learning_rate)

    return output, loss.item()


n_iters = 100_000
print_every = 5000
plot_every = 1000

# 도식화를 위한 손실 추적
current_loss = 0
all_losses = []


def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return f"{m}m {s}s"


start = time.time()
for iter in range(1, n_iters + 1):
    category, line, category_tensor, line_tensor = randomTrainingExample()
    output, loss = train(category_tensor.to(DEVICE), line_tensor.to(DEVICE))
    current_loss += loss

    # iter 숫자 손실 이름 추측 화면 출력
    if iter % print_every == 0:
        guess, guess_i = categoryFromOutput(output)
        correct = '✓' if guess == category else '✗ (%s)' % category
        print('%d %d%% (%s) %.4f %s / %s %s' % (iter, iter / n_iters * 100, timeSince(start), loss, line, guess, correct))
        
    # 현재 평균 손실 -> 전체 손실 리스트 추가
    if iter % plot_every == 0:
        all_losses.append(current_loss / plot_every)
        current_loss = 0


# 결과 평가 confusion matrix를 만듬
# 주축에서 벗어난 밝은 점 -> 잘못 추측한 언어
confusion = torch.zeros(n_categories, n_categories)
n_confution = 10000

# 주어진 라인의 출력 반환
def evaluate(line_tensor):
    hidden = rnn.initHidden().to(DEVICE)

    for i in range(line_tensor.size()[0]):
        output, hidden = rnn(line_tensor[i].to(DEVICE), hidden)
    
    return output

# 예시들 중에 어떤 것이 정확하게 예측되었는지 기록
for i in range(n_confution):
    category, line, category_tensor, line_tensor = randomTrainingExample()
    output = evaluate(line_tensor)
    guess, guess_i = categoryFromOutput(output)
    category_i = all_categories.index(category)
    confusion[category_i][guess_i] += 1

# 모든 행을 합계로 나눠 정규화
for i in range(n_categories):
    confusion[i] = confusion[i] / confusion[i].sum()

fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(confusion.numpy())
fig.colorbar(cax)

# 축 설정
ax.set_xticklabels([''] + all_categories, rotation=90)
ax.set_yticklabels([''] + all_categories)

# 모든 tick에서 레이블 지정
ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

# sphinx_gallery_thumbnail_number = 2
plt.show()

def predict(input_line, n_predictions=3):
    print('\n> %s' % input_line)
    with torch.no_grad():
        output = evaluate(lineToTensor(input_line))

        # Get top N categories
        topv, topi = output.topk(n_predictions, 1, True)
        predictions = []

        for i in range(n_predictions):
            value = topv[0][i].item()
            category_index = topi[0][i].item()
            print('(%.2f) %s' % (value, all_categories[category_index]))
            predictions.append([value, all_categories[category_index]])

predict('Dovesky')
predict('Jackson')
predict('Satoshi')


if __name__ == '__main__':
    ...
    # print(torch.__version__)

    # # RNN
    # rnn = nn.RNN(3, 3)  # Input dim: 3, output dim: 3
    # inputs = [torch.randn(1, 3) for _ in range(5)]   # sequence 길이 5
    # print('inputs:', inputs)
    # hidden = (torch.randn(1, 1, 3))
    # for i in inputs:
    #     out, hidden = rnn(i.view(1, 1, -1), hidden)

    # print('out: ', out)
    # print('out shape: ', out.shape)
    # print('hidden: ', hidden)
    # print('hidden shape: ', hidden.shape)

    # # LSTM
    # lstm = nn.LSTM(3, 3)
    # inputs = [torch.randn(1, 3) for _ in range(5)]
    # print('inputs:', inputs)
    # hidden = (torch.randn(1, 1, 3),
    #         torch.randn(1, 1, 3))
    # for i in inputs:
    #     out, hidden = lstm(i.view(1, 1, -1), hidden)
    # print('out:', out)
    # print('out shape:', out.shape)
    # print('hidden:', hidden)
    # print('hidden state shape:', hidden[0].shape)
    # print('cell state shape:', hidden[1].shape)
    

    # Data에 위치한 각 언어별 txt파일 가져오기
    print(findFiles(CUR_PATH))

    # 문자열을 Ascii 코드로 변경 테스트
    print(unicodeToAscii('Ślusàrski'))

    # 각 언어의 수
    print(n_categories)

    # 이탈리아어 사전 앞 5개
    print(category_lines['Italian'][:5])

    # 원핫 벡터 테스트
    print(letterToTensor('J'))
    print(lineToTensor('Jones').size())

    # RNN output hidden test
    input = letterToTensor('A')
    hidden = torch.zeros(1, n_hidden)

    output, next_hidden = rnn(input, hidden)
    print(output.shape)
    print(hidden.shape)

    input = lineToTensor('Albert')
    hidden = torch.zeros(1, n_hidden)

    output, next_hidden = rnn(input[0], hidden)
    print(output)
    print(output.shape)
    print(next_hidden.shape)
    print(categoryFromOutput(output))

    # 학습 랜덤 샘플링 test
    for i in range(10):
        category, line, catgory_tensor, line_tensor = randomTrainingExample()
        print(f"category = {category} / line = {line}")