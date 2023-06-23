"""
Naive Bayes Classifier 구현

베이지안 통계 가정과 BOW(Bag of Word) 표현 기반 모델
가정
    1. BOW 가정 - 단어의 위치는 확률에 영향을 주지 않는다.
    2. 조건부 독립 - 클래스가 주어지면 속성들의 확률은 독립적이다.
"""

from typing import Tuple, Set
from collections import Counter
import numpy as np


class NaiveBayesClassifier:
    def __init__(self):
        self.exception = list(".,?!'()*\"\\:-")
    
    def make_vocabulary(self, docs, target) -> Tuple[list, Counter]:
        voc = set()
        bigdoc = [Counter()] * self.num_classes
        for doc, t in zip(docs, target):
            for e in self.exception:
                doc = doc.replace(e, "")                
            words = doc.split(' ')
            bigdoc[t].update(words)
            voc = voc | set(words)
        return list(voc), bigdoc
                
    def train(self, X, y):
        assert len(X) == len(y)
        
        self.train_X = np.array(X)
        self.train_y = np.array(y)
        self.num_docs = self.train_X.shape[0]
        self.classes = np.unique(self.train_y, axis=0)
        self.num_classes = self.classes.shape[0]
        self.voc, self.bigdoc = self.make_vocabulary(X, y)
        self.num_words = [
            sum(c.values())
            for c in self.bigdoc
        ]
        self.log_prior = np.zeros((self.num_classes,))
        self.log_likelihood = np.zeros((len(self.voc), self.num_classes))
        for cls in self.classes:
            self.log_prior[cls] = np.log((self.train_y == cls).sum() / self.num_docs + 1)
        
            for i, word in enumerate(self.voc):
                self.log_likelihood[i, cls] = np.log((self.bigdoc[cls][word] + 1) / (self.num_words[cls] + 1))
        
        return self.log_prior, self.log_likelihood, self.voc
                            
    def predict(self, docs):
        preds = []
        for doc in docs:
            scores = np.ones((self.num_classes, ))
            for e in self.exception:
                doc = doc.replace(e, "")
            for word in doc.split(' '):
                if word in self.voc:
                    word_idx = self.voc.index(word)
                    for i in range(self.num_classes):
                        scores[i] *= self.log_likelihood[word_idx, i]
            scores = scores * self.log_prior
            preds.append(np.argmax(scores))
        
        return np.array(preds)