import textpipeline as tpp
import pandas as pd
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from operator import itemgetter
FILE_PATH = '/Users/george/testData/ML_hw2_moviereview.csv'
STOPWORD_PATH = '../../TextMining_study/stopwords/stopword_seoul.txt'

if __name__ == '__main__':

    corpus = pd.read_csv(FILE_PATH)
    corpus.drop(['Unnamed: 0'], axis=1, inplace=True)

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

    start = time.time()
    #documents = pipeline.preprocessing(corpus['review'])
    documents = pipeline.mpprocessing(corpus['review'])
    end1 = time.time()
    #print(f'inside : docs => {len(documents)} \n{documents}\n')
    #print(f'single process: {end1-start:.3f}s') #take 182.584 s
    print(f'multi process {end1-start:.3f}s') #take 56.611s
    print(f'inside : docs => {len(documents)}\n')

    X = documents
    y_flag = round(corpus['rating'].mean())
    y = [1 if rating >= y_flag else 0 for rating in corpus['rating']]



    # if len(X) - len(y) == 0:
    #     print(f'same size of X, y')

    #slpit
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    #vecotrizing
    tfidf_vectorizer = TfidfVectorizer()
    X_train_v = tfidf_vectorizer.fit_transform(X_train)
    X_test_v = tfidf_vectorizer.transform(X_test)

    vocablist = [word for word, _ in sorted(tfidf_vectorizer.vocabulary_.items(), key=lambda x:x[1], reverse=True)]

    text_logi = LogisticRegression(C=0.2, penalty='l2',solver='saga')
    text_logi.fit(X_train_v, y_train)
    pred = text_logi.predict(X_test_v)

    print('Misclassified samples: {} out of {}'.format((pred != y_test).sum(), len(y_test)))
    print(f'Accuracy: {accuracy_score(y_test, pred):.2f}')


    coefficients = text_logi.coef_.tolist()

    sorted_coefficients = sorted(enumerate(coefficients[0]), key=lambda x:x[1], reverse=True)
    print(sorted_coefficients[:5])

    for word, coef in sorted_coefficients[:50]:
        print('{0:} ({1:.3f})'.format(vocablist[word], coef))

    for word, coef in sorted_coefficients[-50:]:
        print('{0:} ({1:.3f})'.format(vocablist[word], coef))