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
        #self.topic = dict(map(lambda x: (x+1, []), range(k)))
        #self.word_allcated = {}
        self.word_allcated = defaultdict()
        self.vocabulary_ = []
        self._positions = None
        self.candidates = []
        self.t2d = None
        self.t2w = None

    def __call__(self):
        print('call')

    def fit(self, X):
        self._candidate(X)
        self._random_allocate_topic()
        self._make_positions(X)
        self.distribution_topicBYdoc()
        self.distribution_topicBYword()
        self.allocate_topic()
        return self.t2w

    def transform(self):
        pass

    def _candidate(self, corpus):
        voca_candi = []
        for doc in corpus:
            voca_candi.extend([d[0] for d in doc])
        self.candidates = list(set(voca_candi))

    def _random_allocate_topic(self):
        for word in self.candidates:
            self.word_allcated[word] = np.random.randint(1, self.k+1)

    def _make_positions(self, corpus):
        positions = []
        for doc in corpus:
            positions.append([self.word_allcated[word[0]] for word, _ in Counter(doc).items()])
        self._positions = np.array(positions, dtype=object)

    def distribution_topicBYdoc(self):
        distributions = []
        for doc in self._positions:
            defalut_counter = defaultdict(lambda: self.a)
            for i in range(self.k):
                defalut_counter[i+1]

            defalut_counter.update(dict((k, v+self.a) for k, v in Counter(doc).items()))
            distributions.append(list(defalut_counter.values()))

        t2d = np.array(distributions)
        self.t2d = np.transpose(t2d)
        return

    def distribution_topicBYword(self):
        distributions = []
        row = [(word, val) for word, val in self.word_allcated.items()]
        for topic in range(self.k):
            distributions.append([word[1] + self.b if word[1]==(topic+1) else self.b for word in row])
        t2w = np.array(distributions)
        self.t2w = t2w
        return

    def allocate_topic(self):
        for word in self.word_allcated.keys():
            self.word_allcated[word] = 0
            a1 = self.t2d[0,0]/(self.t2d[:,0]).sum()
            self.t2w[0,:]

    def _make_vocabulary(self, corpus):

       pass

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

    c.fit(documents)
    candi = c.t2d
    print(f'{type(candi)} {len(candi)}')
    print(candi.shape)
    print(candi[1,:5])
