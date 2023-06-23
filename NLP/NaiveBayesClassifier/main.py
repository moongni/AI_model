import os

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from NavieBayesClassifier import NaiveBayesClassifier

cur_path = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(cur_path, 'spam.csv'), encoding='latin1')
df = df[['v1', 'v2']]

encoder = LabelEncoder()
df['v1'] = encoder.fit_transform(df['v1'])

train_set, test_set = train_test_split(df, test_size=0.2)
print(f"Train set size: {len(train_set)}, Test set size: {len(test_set)}")

nbc = NaiveBayesClassifier()
log_prior, log_likelihood, V = nbc.train(train_set['v2'], train_set['v1'])

print(log_prior)
print(log_likelihood.shape)
print(len(V))

targets = np.array(test_set['v1'])
preds = nbc.predict(test_set['v2'])
accuracy = (targets == preds).sum() / len(targets)
print(preds)
print(f"Test Accuracy: {accuracy * 100:.2f}")