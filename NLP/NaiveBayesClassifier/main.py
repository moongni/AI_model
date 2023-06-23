import os

import numpy as np
import pandas as pd

from NavieBayesClassifier import NaiveBayesClassifier

cur_path = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(cur_path, 'spam.csv'), encoding='latin1')

labels = np.array(df['v1'])
docs = np.array(df['v2'])

labels[labels == 'ham'] = 0
labels[labels == 'spam'] = 1
labels = labels.astype(np.int64)

nbc = NaiveBayesClassifier()
nbc.train(docs, labels)
print(nbc.train_X.shape)
print(nbc.train_y.shape)
print(nbc.num_docs)
print(nbc.num_classes)
print(nbc.classes)
print(nbc.logPrior)