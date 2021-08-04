import numpy as np
import pandas as pd
import time
from functools import reduce
from pipeline import textpipeline as tpp
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

class BaseClassifer:
    def __init__(self):
        self.X = None
        self.feature = None


    def fit(self):
        pass

    def transform(self):
        pass


class TextClassifier(object):
    """
    input :param :

    """
    def __init__(self):
        self.norm = None
        self.adj_matrix = None
        self.vocabulary = None

    def fit(self, X):
        print([' '.split(doc) for doc in X])
        ds = [' '.split(doc) for doc in X]
        self.norm = reduce(lambda l1, l2: l1 + l2, ds)

    def predict(self):
        pass

    def coef_(self):
        pass

    def trucksvd(self):


        pass


if __name__ == "__main__":

    STOPWORD_PATH = '../../TextMining_study/stopwords/stopword_seoul.txt'
    FILE_PATH = '/Users/george/testData/ML_hw2_moviereview.csv'
    corpus = pd.read_csv(FILE_PATH)

    corpus.dropna(inplace=True)
    print(corpus[corpus['rating'].between(4,6)])

    #corpus.drop(corpus[corpus['rating'].between(4,6)], axis=0)

    # for idx, review in enumerate(corpus['review']):
    #     if not isinstance(review, str):
    #         corpus.drop([idx], inplace=True)
    # print(f'corpus: {len(corpus)}')
    # print(corpus.info())



    y = corpus['rating']
    X = corpus['review']
    train_X, train_y, test_X, test_y = train_test_split(X[:100], y[:100], test_size=0.33, random_state=42)

    print(type(train_X))
    pipeline = tpp.PreProcessor([
        ('tokenize', tpp.Tokenizer()),
        ('postag', tpp.PosTaging(name='mecab', stop_pos=['NN*','VV*'])),
        ('stopwords', tpp.StopWordsFilter(stopword_path=STOPWORD_PATH)),
        ('selectword', tpp.Selector(flat=True))
    ])

    start = time.time()
    documents = pipeline.mpprocessing(train_X, 5)
    #documents = pipeline.preprocessing(train_X)
    print(f'multi processs\t{time.time() - start:.3f} time..')

    print(documents)
    model = TextClassifier()
    model.fit(documents)

    print(model.norm)



