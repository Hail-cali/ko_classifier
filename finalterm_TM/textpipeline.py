import nltk.tokenize
import csv
import re
import math
import os
from multiprocessing import Pool
from contextlib import contextmanager
import pandas as pd

class BaseProcessor:
    IN_TYPE = [list]
    OUT_TYPE = [list, tuple]

class BaseEstimator(object):

    def fit(self):
        pass

    def transform(self):
        pass



class PreProcessor(object):

    Dstep_ = {1: 'tokenize', 2: 'Lemmatizer', 3: 'postag', 4: 'stopwords', 5: 'imputer', 6: 'selectword'}
    def __init__(self, step):
        self.corpus = None
        self.step = step

    def preprocessing(self, corpus):
        check = {}
        if isinstance(corpus, list):
            self.corpus = corpus
        elif isinstance(corpus, str):
            self.corpus = [corpus]
        else:
            self.corpus = [d for d in corpus]

        for step in self.step:
            if isinstance(step[1], object):
                # for docs in self.corpus:
                #     result.append(step[1](docs))
                self.corpus = [step[1](docs) for docs in self.corpus]
                check[step[0]] = True

            else:
                check[step[0]] = False
                print(f'check step: {step[0]}')

        print(check)
        return self.corpus


    def mpprocessing(self, corpus, work_num=4):
        """
        before running
        warp main code with __name__== '__main__'
        """
        check = {}
        if isinstance(corpus, list):
            self.corpus = corpus
        elif isinstance(corpus, str):
            self.corpus = [corpus]
        else:
            self.corpus = [d for d in corpus]
        print(f'work_num : {work_num}')
        q = []
        with Pool(work_num) as p:
            q.append(p.map(self.worker, self.corpus))
        # with self.manager(processes=3) as p:
        #     p.map(self.worker, self.corpus)
        self.corpus = q[0]

        return self.corpus

    @contextmanager
    def manager(self, *args, q):
        pool = Pool(*args, q)
        yield pool
        pool.terminate()

    def worker(self, corpus):
        data = corpus
        for inst in self.step:
            data = inst[1](data)
        return data

class Tokenizer(object):

    def __call__(self, corpus=None):
        if corpus:
            return nltk.word_tokenize(corpus)

class PosTaging(object):

    def __init__(self, name='komoran', stop_pos=['NN*'], mecab_path='/usr/local/lib/mecab/dic/mecab-ko-dic'):
        import konlpy.tag
        pos_str = ''
        for pos in stop_pos:
            if pos.endswith('*'):
                pos_str += '|%s' % pos
            else:
                pos_str += '|%s' % pos

        pos_str = f'[%s]' % pos_str[1:]
        #print(f'pos_str : {pos_str}')
        self.stoppos = re.compile(pos_str)

        if name == 'komoran':
            self.postag = konlpy.tag.Komoran()
        elif name == 'mecab':
            try:
                self.postag = konlpy.tag.Mecab(dicpath=mecab_path)
            except:
                print(f'check mecab path: {mecab_path}')
        elif name == 'okt':
            self.postag = konlpy.tag.Okt()

    def __call__(self, *args):
        try:
            #print(len(args[0]))
            result = list(self.postag.pos(docs)[0] for docs in args[0])
            if isinstance(self.stoppos, object):
                pass
            result = [pos for pos in result if self.stoppos.match(pos[1])]
            return result
        except:
            return []

class StopWordsFilter(object):

    def __init__(self, stopword_path='../stopwords/mystopwords.txt'):
        self.stopword = [line.strip() for line in open(stopword_path, 'r', encoding='utf-8')]

    def __call__(self, corpus=None):
        return self.stop_word_filtering(corpus)

    def stop_word_filtering(self, *args):
        result = []
        for docs in args:
            result.extend([word for word in docs if word[0] not in self.stopword])
        return result

class Selector(object):
    def __init__(self, flat=False):
        self.flattend = flat

    def __call__(self, corpus=None):
        if self.flattend:
            return ' '.join(word[0] for word in corpus)
        return [word[0] for word in corpus]

class TextImputer(object):
    def __init__(self):
        self._freq = -1

    def __call__(self, corpus):
        self._most_frequent(corpus)
        return corpus

    def _del_nan(self, *args):

        pass

    def _most_frequent(self, *args):

        self.freq = 1
        pass

    @property
    def freq(self, val):
        self._freq = val

    @freq.getter
    def feq(self):
        return self.freq

class Stemmer(object):
    pass

class myLemmatizer(object):

    def __init__(self):
        from soylemma import Lemmatizer

        self.inst = Lemmatizer()

    def __call__(self, *args):
        docs = []

        for word in args[0]:

            if not re.findall(r'[NN*]', word[1]):
                lem = self.inst.lemmatize(word[0])
                print(f'raw {word}')
                # if re.findall(r'[V*]', lem[1]):
                #     docs.append(lem)
                print(f'candidate{lem}')
                docs.append(lem[0])
            else:
                docs.append(word)
        #print(docs)
        return docs

class Ngram(Tokenizer):

    def __init__(self, N):
        super().__init__()
        self.N = N


    pass


# def textReader(file_path, filename):
#     corpus = []
#     file = os.path.join(file_path, filename)
#     with open(file, newline='', encoding='utf-8') as file:
#         reader = csv.reader(file, delimiter=',')
#         next(reader)
#         for idx, sample in enumerate(reader):
#
#             corpus.append([idx]+sample[1:])
#
#         return corpus

class TextReader():

    def __init__(self, file, doc_index=2, meta_index=1, delimiter=','):
        self.pair_map = {}
        array = []
        id = 0
        pair_map = {}
        with open(file, encoding='utf-8') as ins:
            reader = csv.reader(ins)
            next(reader)

            for fields in reader:
                try:
                    if len(fields[doc_index]) > 35:

                        array.append(fields[doc_index])
                        pair_map[id] = fields[meta_index][:4]

                        id += 1
                except IndexError:
                    print("out of index " + str(id))

        self.docs = array
        self.pair_map = pair_map

    def _categorizer(self):
        self.pair_map

        return

    def __iter__(self):
        return self.docs.__iter__()

    def __len__(self):
        return self.docs.__len__()