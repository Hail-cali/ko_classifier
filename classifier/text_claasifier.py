import numpy as np
from pipeline import textpipeline as tpp
import pandas as pd
import time
from sklearn.model_selection import train_test_split

class BaseClassifer:
    def __init__(self):
        self.X = None
        self.feature = None


    def fit(self):
        pass

    def transform(self):
        pass


class TextClassifier(object):

    def __init__(self):
        pass

    def fit(self, corpus):
        pass

    def predict(self):
        pass

    def coef_(self):
        pass

    def importent_word_(self):
        pass


if __name__ == "__main__":

    STOPWORD_PATH = '../../TextMining_study/stopwords/stopword_seoul.txt'
    FILE_PATH = '/Users/george/testData/ML_hw2_moviereview.csv'
    df = pd.read_csv(FILE_PATH)
    print(df.columns)
    y = df.rating
    X = df.review
    train_X, train_y, test_X, test_y = train_test_split(X, y, test_size=0.33, random_state=42)

    pipeline = tpp.PreProcessor([
        ('tokenize', tpp.Tokenizer()),
        ('postag', tpp.PosTaging(name='mecab', stop_pos=['NN*'])),
        ('stopwords', tpp.StopWordsFilter(stopword_path=STOPWORD_PATH)),
        ('selectword', tpp.Selector(flat=True))
    ])
    start = time.time()
    documents = pipeline.mpprocessing(train_X, 5)
    print(f'multi processs\t{time.time() - start:.3f} time..')




