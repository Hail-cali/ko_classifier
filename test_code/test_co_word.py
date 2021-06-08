import os
import treform as ptm
import time
from pipeline import textpipeline as tpp

#FILE_PATH = './seoul_city_complaints_total_2.csv'
FILE_PATH = '/Users/george/testData/seoul_city_complaints_total_2.csv'
STOPWORD_PATH = '/Users/george/testData/stopwords/stopwords_seoul_210601.txt'
MECAB_PATH = '/Users/george/package_data/mecab-0.996-ko-0.9.2/mecab-ko-dic-2.1.1-20180720/'

if __name__ == '__main__':

    corpus = tpp.TextReader(FILE_PATH, doc_index=(3,4), meta_index=2)

    pipeline = tpp.PreProcessor([
        ('tokenize', tpp.Tokenizer()),
        ('postag', tpp.PosTaging(name='mecab', stop_pos=['NN*'], mecab_path=MECAB_PATH)),
        # ('lemmatizer', tpp.myLemmatizer()),
        ('stopwords', tpp.StopWordsFilter(stopword_path=STOPWORD_PATH)),
        ('selectword', tpp.Selector(flat=True))
    ])

    start = time.time()
    documents = pipeline.mpprocessing(corpus, 5)
    end1 = time.time()
    print(f'multi process {end1 - start:.3f}s')
    print(documents)
    #pair_map = corpus.pair_map

    result = documents

    with open('../test_data/processed_seoul.txt', 'w', encoding='utf-8') as f_out:
        for doc in result:
            f_out.write(doc + "\n")
    f_out.close()

    input_file = '../test_data/processed_seoul.txt'
    output_file = 'co_seoul_2015.txt'
    worker_number = 3
    threshold_value = 5

    program_path = '/Users/george/PycharmProjects/TextMining_study/external_programs/'

    if os.path.isdir(program_path) != True:
        raise Exception(program_path + ' is not a directory')

    co_occur = ptm.cooccurrence.CooccurrenceExternalManager(program_path=program_path, input_file=input_file,
                                                            output_file=output_file, threshold=threshold_value,
                                                            num_workers=worker_number)

    co_occur.execute()

    co_results = {}
    vocabulary = {}
    word_hist = {}
    with open(output_file, 'r', encoding='utf-8') as f_in:
        for line in f_in:
            fields = line.split()
            token1 = fields[0]
            token2 = fields[1]
            token3 = fields[2]

            if token2 == token1:
                continue

            tup = (str(token1), str(token2))
            co_results[tup] = float(token3)

            vocabulary[token1] = vocabulary.get(token1, 0) + 1
            vocabulary[token2] = vocabulary.get(token2, 0) + 1

            word_hist = dict(zip(vocabulary.keys(), vocabulary.values()))

    graph_builder = ptm.graphml.GraphMLCreator()

    # mode is either with_threshold or without_threshod
    mode = 'with_threshold'

    if mode == 'without_threshold':
        graph_builder.createGraphML(co_results, vocabulary.keys(), "test_seoul.graphml")

    elif mode == 'with_threshold':
        graph_builder.createGraphMLWithThresholdInDictionary(co_results, word_hist, "test_seoul.graphml", threshold=10.0)
        display_limit = 20
        graph_builder.summarize_centrality(limit=display_limit)