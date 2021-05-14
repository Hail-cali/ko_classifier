import numpy as np
import nltk.tokenize
import pandas as pd
import csv
import re


FILE_PATH = '/Users/george/testData/'
STOPWORD_PATH = '../../TextMining_study/stopwords/stopword_seoul.txt'

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

        if name =='komoran':
            self.postag = konlpy.tag.Komoran()
        elif name =='mecab':
            try:
                self.postag = konlpy.tag.Mecab(dicpath=mecab_path)
            except:
                print(f'check mecab path: {mecab_path}')


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

        return corpus

corpus = pd.read_csv('/Users/george/testData/seoul_city_complaints_2019_2021.csv')
print(corpus['ask'][150])
print('-'*30)

#tt = textReader(FILE_PATH, 'seoul_city_complaints_2019_2021.csv')

en = '''Edgar Allan Poe's C. Auguste Dupin is generally acknowledged as the first detective in fiction and served as the
 prototype for many later characters, including Holmes.[7] Conan Doyle once wrote, "Each [of Poe's detective stories] is
  a root from which a whole literature has developed... Where was the detective story until Poe breathed the breath of 
  life into it?"[8] Similarly, the stories of Émile Gaboriau's Monsieur Lecoq were extremely popular at the time Conan 
  Doyle began writing Holmes, and Holmes's speech and behaviour sometimes follow that of Lecoq.[9][10] Doyle has his 
  main characters discuss these literary antecedents near the beginning of A Study in Scarlet, which is set soon after 
  Watson is first introduced to Holmes. Watson attempts to compliment Holmes by comparing him to Dupin, to which Holmes 
  replies that he found Dupin to be "a very inferior fellow" and Lecoq to be "a miserable bungler".[11]'''

#corpus pipeline start
pipeline = PreProcessor([
    ('tokenize', Tokenizer()),
    ('postag', PosTaging(name='mecab')),
    ('stopwords', StopWordsFilter(stopword_path=STOPWORD_PATH))
    ])
# a = Tokenizer()
# out_rst = a(en)
# print(f'outside : sentence => {len(out_rst)} \n{out_rst}\n')

in_rst = pipeline.preprocessing(corpus['ask'][150:152])
print(f'inside : docs => {len(in_rst)} \n{in_rst}\n')

