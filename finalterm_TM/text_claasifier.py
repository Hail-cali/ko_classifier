import numpy as np
from sklearn.pipeline import Pipeline
import nltk.tokenize
import pandas as pd
import csv

file_path = '/Users/george/testData/'

def textReader(file_path, filename):
    corpus = []
    with open(file_path + filename, newline='', encoding='utf-8') as file:
        reader = csv.reader(file, delimiter=',')
        next(reader)
        for idx, sample in enumerate(reader):

            corpus.append([idx]+sample[1:])

        return corpus

class PreProcessor(object):

    def __init__(self, step):
        self.corpus = None
        self.step = step

    def preprocessing(self, corpus):
        result = []
        if isinstance(corpus, list):
            self.corpus = corpus
        else:
            self.corpus = [d for d in corpus]

        for step in self.step:
            if isinstance(step[1], object):
                for docs in corpus:
                    result.append(step[1](docs))
            else:
                print(f'check step: {step[0]}')

class TokenizerEN(object):

    def __call__(self, corpus=None):
        if corpus:
            return nltk.word_tokenize(corpus)


corpus = pd.read_csv('/Users/george/testData/seoul_city_complaints_2019_2021.csv')
print(corpus['ask'][150])
print('-'*30)
tt = textReader(file_path, 'seoul_city_complaints_2019_2021.csv')

p = PreProcessor([
    ('tokenize', TokenizerEN())
    ])
a = TokenizerEN()

# print(f'{type(p)}: {isinstance(p, object)}')
# print(f'{type(corpus)}: {isinstance(corpus, type)}')
# print(f'{type(tt)}: {isinstance(tt, type)}')
print(p.preprocessing())