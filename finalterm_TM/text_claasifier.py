import numpy as np
from collections import defaultdict

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

class TextClusutering(object):

    def __init__(self, k=5):
        self.k = k
        self.doc2vec = defaultdict()
        self.adj = []
        self.vectorizer = False
        self.X = False

    def __call__(self):
        pass

    def process(self, corpus):

        return corpus

    def make_matrix(self, *args):


        return