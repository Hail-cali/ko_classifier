import textpipeline as tpp
import pandas as pd
import numpy as np
import time
from numpy.linalg import svd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

FILE_PATH = '/Users/george/testData/seoul_data/dml_seoul_city_complaints_2020_2021.csv'
#STOPWORD_PATH = '../../TextMining_study/stopwords/stopword_seoul.txt'
STOPWORD_PATH = '/Users/george/testData/stopwords/stopwords_19_complaints.txt'

if __name__ == '__main__':
    corpus = pd.read_csv(FILE_PATH)

    #corpus pipeline start
    pipeline = tpp.PreProcessor([
        ('tokenize', tpp.Tokenizer()),
        ('postag', tpp.PosTaging(name='mecab', stop_pos=['NN*'])),
        #('lemmatizer', tpp.myLemmatizer()),
        ('stopwords', tpp.StopWordsFilter(stopword_path=STOPWORD_PATH)),
        ('selectword', tpp.Selector(word=True))
        ])

    start2 = time.time()
    documents_mp = pipeline.mpprocessing(corpus['complaints'], 5)
    #documents_mp = pipeline.mpprocessing(corpus['answers'],5)
    end2 = time.time()
    print(f'multi process\t{end2-start2:.3f}')

    #make DTM
    cv = CountVectorizer()
    DTM = cv.fit_transform(documents_mp).toarray()
    feature_names = cv.get_feature_names()
    idx_voca = cv.vocabulary_
    print(feature_names)
    print(DTM)
    print(f'type of DTM {type(DTM)} shape of DTM {DTM.shape}')

    #svd
    # U, sigma, Vt = svd(DTM)
    # print(f'U {U.shape} sigma {sigma.shape} Vt {Vt.shape}')
    # print(sigma[:10])
    svd = TruncatedSVD(n_components=5, n_iter=7, random_state=42)
    X = svd.fit(DTM)

    print(svd.components_)
    print(type(svd.components_))
    print(len(svd.components_))
    best_ask = [feature_names[i] for i in svd.components_[0].argsort()[::-1]]
    print('-'*10)
    print(best_ask[:30])