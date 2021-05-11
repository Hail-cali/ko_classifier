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
        elif isinstance(corpus,str):
            self.corpus = [corpus]
        else:
            self.corpus = [d for d in corpus]

        for step in self.step:
            if isinstance(step[1], object):
                for docs in self.corpus:
                    result.append(step[1](docs))
            else:
                print(f'check step: {step[0]}')
        return result

class TokenizerEN(object):

    def __call__(self, corpus=None):
        if corpus:
            return nltk.word_tokenize(corpus)


corpus = pd.read_csv('/Users/george/testData/seoul_city_complaints_2019_2021.csv')
print(corpus['ask'][150])
print('-'*30)
tt = textReader(file_path, 'seoul_city_complaints_2019_2021.csv')
en = '''Edgar Allan Poe's C. Auguste Dupin is generally acknowledged as the first detective in fiction and served as the
 prototype for many later characters, including Holmes.[7] Conan Doyle once wrote, "Each [of Poe's detective stories] is
  a root from which a whole literature has developed... Where was the detective story until Poe breathed the breath of 
  life into it?"[8] Similarly, the stories of Émile Gaboriau's Monsieur Lecoq were extremely popular at the time Conan 
  Doyle began writing Holmes, and Holmes's speech and behaviour sometimes follow that of Lecoq.[9][10] Doyle has his 
  main characters discuss these literary antecedents near the beginning of A Study in Scarlet, which is set soon after 
  Watson is first introduced to Holmes. Watson attempts to compliment Holmes by comparing him to Dupin, to which Holmes 
  replies that he found Dupin to be "a very inferior fellow" and Lecoq to be "a miserable bungler".[11]'''
p = PreProcessor([
    ('tokenize', TokenizerEN())
    ])
a = TokenizerEN()

out_rst = a(en)
print(f'outside : sentence => {len(out_rst)} \n{out_rst}\n')

in_rst = p.preprocessing(corpus['ask'])
print(f'inside : sentence => {len(in_rst)} \n{in_rst}\n')