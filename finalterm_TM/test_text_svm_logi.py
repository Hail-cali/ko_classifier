import textpipeline as tpp
import pandas as pd
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

FILE_PATH = '/Users/george/testData/ML_hw2_moviereview.csv'
STOPWORD_PATH = '../../TextMining_study/stopwords/stopword_seoul.txt'

corpus = pd.read_csv(FILE_PATH)
corpus.drop(['Unnamed: 0'],axis=1, inplace=True)

#temp NaN processing,
print(f'corpus: {len(corpus)}')
for idx, review in enumerate(corpus['review']):
    if not isinstance(review, str):
        print(review)
        corpus.drop([idx],inplace=True)
print(f'corpus: {len(corpus)}')
print(corpus.info())



pipeline = tpp.PreProcessor([
    ('tokenize', tpp.Tokenizer()),
    ('postag', tpp.PosTaging(name='mecab', stop_pos=['NN*', 'V*'])),
    ('stopwords', tpp.StopWordsFilter(stopword_path=STOPWORD_PATH)),
    ('selectword', tpp.Selector(word=True))
    ])


documents = pipeline.preprocessing(corpus['review'][1:3])
print(f'inside : docs => {len(documents)} \n{documents}\n')
print(f"{corpus['rating'].mean()}")
X = pd.DataFrame(documents, columns=['token_r'])

y_flag = round(corpus['rating'].mean())
y = pd.DataFrame([1 if rating >= y_flag else 0 for rating in corpus['rating'] ], columns=['bi_rating'])

#
# if len(X) - len(y) == 0:
#     print(f'same size of X, y')
#
#
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# tf_idf_vectorizer = TfidfVectorizer()
# X_v = tf_idf_vectorizer.fit_transform(X_train)
# print(X_v)

