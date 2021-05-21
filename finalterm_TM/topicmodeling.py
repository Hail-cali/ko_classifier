import numpy as np
import textpipeline as tpp
import pandas as pd
import time
from collections import defaultdict
from collections import Counter
class BaseLDA:
    IN_TYPE = [list, tuple]
    OUT_TYPE = [list, str]

class TopicModeling(object):

    def __init__(self, a=0.01, b=0.001, k=5):
        self.a = a # 문서들의 토픽 분포를 얼마나 밀집되게 할 것인지
        self.b = b # 문서 내 단어들의 토픽 분포를 얼마나 밀집되게 할 것인지
        self.k = k #몇개의 토픽으로 구성할 건지
        self._X = None
        self.topic = dict(map(lambda x: (x+1, []), range(k)))
        self.word_allcated = {}
        self._vocabulary = []
        self.positions = []
        self.candidates = []
        self.result = []

    def __call__(self):
        print('call')

    def fit(self, X):
        self._candidate(X)
        self._random_allocate_topic()
        result = self.distribution_topicBYdoc()
        return result

    def transform(self):
        pass

    def make_vocabulary(self, corpus):

       pass



    def _candidate(self, corpus):

        voca_candi = []
        for doc in corpus:
            voca_candi.extend([d[0] for d in doc])
        self.candidates = list(set(voca_candi))



        for doc in corpus:
            a = list(set(map(lambda x: x[0], doc)))
            self.result.append(a)


    def _random_allocate_topic(self):
        for word in self.candidates:
            self.word_allcated[word] = np.random.randint(1, self.k+1)



        for doc in self.result:
            self.positions.append(np.random.randint(1, self.k+1, size=len(doc)))


    def distribution_topicBYdoc(self):
        topicBYdoc = []



        topicBydoc = []
        for word_topic in self.positions:
            topicBydoc.append([count +self.a for _, count in sorted(Counter(word_topic).items(), key= lambda x: x[0])])

        return topicBydoc

    def _svm(self):
        pass

    @property
    def X(self, *args):
        self._X = args

    @X.getter
    def X(self):
        return self._X

FILE_PATH = '/Users/george/testData/'
STOPWORD_PATH = '../../TextMining_study/stopwords/stopword_seoul.txt'

if __name__ =='__main__':

    c = TopicModeling()

    df = pd.read_csv('~/testData/seoul_city_complaints_2019_2021.csv')

    pipeline = tpp.PreProcessor([
        ('tokenize', tpp.Tokenizer()),
        ('postag', tpp.PosTaging(name='mecab', stop_pos=['NN*'])),
        ('stopwords', tpp.StopWordsFilter(stopword_path=STOPWORD_PATH)),
        #('selectword', tpp.Selector(word=True))
    ])

    start = time.time()
    documents = pipeline.mpprocessing(df['ask'])
    print(f'multi processs\t{time.time() - start:.3f} time..')

    candi = c.fit(documents)
    print(f'{type(candi)} {len(candi)}')
    print(f'{type(candi[1])} {len(candi[1])}')
    print(f'{candi[1]}')
    temp = c.positions
    print(f'{type(Counter(temp[1]))} {Counter(temp[1])}')
    print([count+0.01 for _, count in sorted(Counter(temp[1]).items(), key= lambda x: x[0])])
    print(c.word_allcated)