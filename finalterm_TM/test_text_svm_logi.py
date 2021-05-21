import textpipeline as tpp
import pandas as pd
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from operator import itemgetter
from sklearn.model_selection import GridSearchCV
from imblearn.over_sampling import SMOTE


import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, roc_auc_score

plt.style.use('seaborn')

FILE_PATH = '/Users/george/testData/ML_hw2_moviereview.csv'
STOPWORD_PATH = '../../TextMining_study/stopwords/stopword_seoul.txt'

if __name__ == '__main__':
    #data process mode
    work = 'multi'
    #read data
    corpus = pd.read_csv(FILE_PATH)
    corpus.drop(['Unnamed: 0'], axis=1, inplace=True)

    #temp NaN processing,
    print(f'corpus: {len(corpus)}')
    for idx, review in enumerate(corpus['review']):
        if not isinstance(review, str):
            corpus.drop([idx],inplace=True)
    print(f'corpus: {len(corpus)}')
    print(corpus.info())

    #pipeline inst
    pipeline = tpp.PreProcessor([
        ('tokenize', tpp.Tokenizer()),
        ('postag', tpp.PosTaging(name='mecab', stop_pos=['NN*', 'VV*'])),
        ('stopwords', tpp.StopWordsFilter(stopword_path=STOPWORD_PATH)),
        ('selectword', tpp.Selector(word=True))
        ])

    #processing step
    if work == 'single':
        start = time.time()
        documents = pipeline.preprocessing(corpus['review'])
        end1 = time.time()
        print(f'single process: {end1-start:.3f}s') #take 182.584 s

    elif work == 'multi':
        start = time.time()
        documents = pipeline.mpprocessing(corpus['review'],5)
        end1 = time.time()
        print(f'multi process {end1-start:.3f}s') #take 56.611s

    # print(f'inside : docs => {len(documents)} \n{documents}\n')
    print(f'inside : docs => {len(documents)}\n')

    X = documents
    y_flag = round(corpus['rating'].mean())
    y = [1 if rating >= y_flag else 0 for rating in corpus['rating']]
    y_new = []
    X_new = []

    for idx, label in enumerate(corpus['rating']):
        if label >= y_flag:
            y_new.append(1)
            X_new.append(X[idx])
        elif label >= 5:
            continue
        else:
            y_new.append(0)
            X_new.append(X[idx])


    if len(X_new) - len(y_new) == 0:
        print(f'same size of X({len(X_new)}) y({len(y_new)})')

    #slpit
    X_train, X_test, y_train, y_test = train_test_split(X_new, y_new, test_size=0.2, random_state=42)

    #vecotrizing
    tfidf_vectorizer = TfidfVectorizer()  # 0.90  # 65.774s
    #tfidf_vectorizer = CountVectorizer()   # 0.90  # 120s

    X_train_v = tfidf_vectorizer.fit_transform(X_train)
    X_test_v = tfidf_vectorizer.transform(X_test)

    vocablist = [word for word, _ in sorted(tfidf_vectorizer.vocabulary_.items(), key=lambda x:x[1], reverse=True)]

    # imbalance data sampling using SMOTE
    sm = SMOTE(random_state=0) # 10, l2, sag => 0.78
    X_resampled, y_resampled = sm.fit_resample(X_train_v, y_train)

    text_logi = LogisticRegression() # 0.2 l2, saga => 0.90
    param = {'C': [0.1, 0.2, 1, 2, 10], 'penalty': ['l2'], 'solver': ['saga', 'sag']}
    gsv = GridSearchCV(text_logi, param_grid=param, n_jobs=5)
    #gsv.fit(X_train_v, y_train)
    gsv.fit(X_resampled, y_resampled)

    best_param = gsv.best_params_
    print(f'best param: {best_param}')

    pred = gsv.predict(X_test_v)

    print(f'Misclassified samples: {(pred != y_test).sum()} out of {len(y_test)}')
    print(f'Accuracy: {accuracy_score(y_test, pred):.2f}')

    best_logi = LogisticRegression(C=best_param['C'], penalty=best_param['penalty'], solver=best_param['solver'])

    #text_logi.fit(X_train_v, y_train)
    #pred = text_logi.predict(X_test_v)
    #coefficients = text_logi.coef_.tolist()

    #best_logi.fit(X_train_v, y_train)
    best_logi.fit(X_resampled, y_resampled)
    coefficients = best_logi.coef_.tolist()

    sorted_coefficients = sorted(enumerate(coefficients[0]), key=lambda x: x[1], reverse=True)
    print(sorted_coefficients[:5])

    for word, coef in sorted_coefficients[:50]:
        print(f'{vocablist[word]} {coef:.3f}')
        #print('{0:} ({1:.3f})'.format(vocablist[word], coef))

    for word, coef in sorted_coefficients[-50:]:
        print(f'{vocablist[word]} {coef:.3f}')
        #print('{0:} ({1:.3f})'.format(vocablist[word], coef))


    fpr, tpr, threst = roc_curve(y_test, pred, pos_label=1)
    random_prob = [0 for i in range(len(y_test))]
    p_fpr, p_tpr, _ = roc_curve(y_test, random_prob, pos_label=1)
    auc_socore = roc_auc_score(y_test, pred)
    print(f'auc_score {auc_socore}')

    plt.plot(fpr, tpr, linestyle='--', color='orange', label='Logistic Regression')
    plt.plot(p_fpr, p_tpr, linestyle='--', color='blue')
    plt.title('ROC curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc='best')
    plt.savefig('ML_HW2roc_2.png', dpi=300)
    plt.show()