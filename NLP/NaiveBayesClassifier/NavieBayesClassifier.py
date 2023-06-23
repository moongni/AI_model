"""
Naive Bayes Classifier 구현

베이지안 통계 가정과 BOW(Bag of Word) 표현 기반 모델
가정
    1. BOW 가정 - 단어의 위치는 확률에 영향을 주지 않는다.
    2. 조건부 독립 - 클래스가 주어지면 속성들의 확률은 독립적이다.
"""


import numpy as np


class NaiveBayesClassifier:
    def __init__(self):
        ...
        
    def train(self, X, y):
        self.train_X = np.array(X)
        self.train_y = np.array(y)
        self.num_docs = self.train_X.shape[0]
        self.classes = np.unique(self.train_y, axis=0)
        self.num_classes = self.classes.shape[0]
        self.logPrior = np.zeros((self.num_classes,))
        for cls in self.classes:
            print((self.train_y == cls).sum())
            self.logPrior[cls] = np.log((self.train_y == cls).sum() / self.num_docs)
        ...
        
    def eval(self,):
        ...
    

if __name__ == "__main__":
    nbc = NaiveBayesClassifier()
    nbc.train([x for x in range(10)], [y for y in range(10)])
    