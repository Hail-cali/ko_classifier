import textpipeline as tpp
import pandas as pd
import time

from sklearn.feature_extraction.text import TfidfVectorizer
FILE_PATH = '/Users/george/testData/'
STOPWORD_PATH = '../../TextMining_study/stopwords/stopword_seoul.txt'


corpus = pd.read_csv('/Users/george/testData/seoul_city_complaints_2019_2021.csv')
print(corpus['ask'][150])
print('-'*30)

#tt = myp.textReader(FILE_PATH, 'seoul_city_complaints_2019_2021.csv')

en = '''Edgar Allan Poe's C. Auguste Dupin is generally acknowledged as the first detective in fiction and served as the
 prototype for many later characters, including Holmes.[7] Conan Doyle once wrote, "Each [of Poe's detective stories] is
  a root from which a whole literature has developed... Where was the detective story until Poe breathed the breath of 
  life into it?"[8] Similarly, the stories of Émile Gaboriau's Monsieur Lecoq were extremely popular at the time Conan 
  Doyle began writing Holmes, and Holmes's speech and behaviour sometimes follow that of Lecoq.[9][10] Doyle has his 
  main characters discuss these literary antecedents near the beginning of A Study in Scarlet, which is set soon after 
  Watson is first introduced to Holmes. Watson attempts to compliment Holmes by comparing him to Dupin, to which Holmes 
  replies that he found Dupin to be "a very inferior fellow" and Lecoq to be "a miserable bungler".[11]'''

#corpus pipeline start
pipeline = tpp.PreProcessor([
    ('tokenize', tpp.Tokenizer()),
    ('postag', tpp.PosTaging(name='mecab', stop_pos=['NN*'])),
    ('stopwords', tpp.StopWordsFilter(stopword_path=STOPWORD_PATH)),
    ('selectword', tpp.Selector(word=True))
    ])
# a = Tokenizer()
# out_rst = a(en)
# print(f'outside : sentence => {len(out_rst)} \n{out_rst}\n')

start = time.time()
documents = pipeline.preprocessing(corpus['ask'])
print(f'\t{time.time()-start:.3f} time..')
print(f'inside : docs => {len(documents)} \n{documents}\n')


tf_idf_vectorizer = TfidfVectorizer()
X = tf_idf_vectorizer.fit_transform(documents)
print(X)