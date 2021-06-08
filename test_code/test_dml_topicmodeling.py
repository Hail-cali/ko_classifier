import tomotopy as tp
from pipeline import textpipeline as tpp
from clustering.pytext_lda_dmr import pyTextMinerTopicModel
import time


FILE_PATH = './seoul_city_complaints_total_2.csv'
STOPWORD_PATH = '/Users/george/testData/stopwords/stopwords_seoul_210601.txt'
MECAB_PATH = '/Users/george/package_data/mecab-0.996-ko-0.9.2/mecab-ko-dic-2.1.1-20180720/'
if __name__ == '__main__':

    corpus = tpp.TextReader(FILE_PATH, doc_index=(3,4), meta_index=2)

    pipeline = tpp.PreProcessor([
        ('tokenize', tpp.Tokenizer()),
        ('postag', tpp.PosTaging(name='mecab', stop_pos=['NN*'],mecab_path=MECAB_PATH)),
        # ('lemmatizer', tpp.myLemmatizer()),
        ('stopwords', tpp.StopWordsFilter(stopword_path=STOPWORD_PATH)),
        ('selectword', tpp.Selector(flat=False))
    ])

    start = time.time()
    documents = pipeline.mpprocessing(corpus, 5)
    #documents = pipeline.preprocessing(corpus)
    end1 = time.time()
    print(f'multi process {end1 - start:.3f}s')

    pair_map = corpus.pair_map

    text_data = []
    for d in documents:
        if len(d) > 1:
            text_data.append(d)

    topic_model = pyTextMinerTopicModel()
    topic_number = 10

    mdl=None

    #mode='visualize'
    mode = 'dmr'
    label = ''
    if mode == 'lda':
        print('Running LDA')
        label = 'LDA'
        lda_model_name = './test.lda.bin'
        mdl=topic_model.lda_model(text_data, lda_model_name, topic_number)
        print(type(mdl.doc))
        print('perplexity score ' + str(mdl.perplexity))

        labeled_data = []

    elif mode == 'dmr':
        print('Running DMR')
        label='DMR'
        dmr_model_name='./test.dmr.bin'
        mdl=topic_model.dmr_model(text_data, pair_map, dmr_model_name, topic_number)
        print('perplexity score ' + str(mdl.perplexity))

    if mode == 'visualize':
        model_name = './test.lda.bin'
        if model_name == './test.lda.bin':
            mdl = tp.LDAModel.load(model_name)

        mdl.load(model_name)

        v_file_name = FILE_PATH[-11:-6]
        visualization_file='../topic_visualization_' + v_file_name + str(topic_number) + '.html'
        fff = 'topic_visualization_total.html'
        #topic_model.make_pyLDAVis(mdl, visualization_file=visualization_file)
        topic_model.make_pyLDAVis(mdl, visualization_file=fff)