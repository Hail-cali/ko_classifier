import sys
import tomotopy as tp
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class pyTextMinerTopicModel:
    def __init__(self):
        self.name = "Topic Model"

    def make_pyLDAVis(self, mdl, visualization_file='./visualization.html'):
        import pyLDAvis
        topic_term_dists = np.stack([mdl.get_topic_word_dist(k) for k in range(mdl.k)])
        doc_topic_dists = np.stack([doc.get_topic_dist() for doc in mdl.docs])
        doc_topic_dists /= doc_topic_dists.sum(axis=1, keepdims=True)
        doc_lengths = np.array([len(doc.words) for doc in mdl.docs])
        vocab = list(mdl.used_vocabs)
        term_frequency = mdl.used_vocab_freq

        prepared_data = pyLDAvis.prepare(
            topic_term_dists,
            doc_topic_dists,
            doc_lengths,
            vocab,
            term_frequency
        )
        pyLDAvis.save_html(prepared_data, visualization_file)

        #pyLDAvis.save_html(vis_data, visualization_file)

    def lda_model(self, text_data, save_path, topic_number=20):
        mdl = tp.LDAModel(tw=tp.TermWeight.ONE, min_cf=3, rm_top=5, k=topic_number)
        index=0
        for doc in text_data:
            print(str(index) + " : " + str(doc))
            mdl.add_doc(doc)
            index+=1

        mdl.burn_in = 100
        mdl.train(0)
        print('Num docs:', len(mdl.docs), ', Vocab size:', mdl.num_vocabs, ', Num words:', mdl.num_words)
        print('Removed top words:', mdl.removed_top_words)
        print('Training...', file=sys.stderr, flush=True)
        for i in range(0, 1500, 10):
            mdl.train(10)
            print('Iteration: {}\tLog-likelihood: {}'.format(i, mdl.ll_per_word))

        print('Saving...', file=sys.stderr, flush=True)
        mdl.save(save_path, True)

        # extract candidates for auto topic labeling
        extractor = tp.label.PMIExtractor(min_cf=10, min_df=5, max_len=5, max_cand=10000)
        cands = extractor.extract(mdl)

        # ranking the candidates of labels for a specific topic
        labeler = tp.label.FoRelevance(mdl, cands, min_df=5, smoothing=1e-2, mu=0.25)
        for k in range(mdl.k):
            print("== Topic #{} ==".format(k))
            print("Labels:", ', '.join(label for label, score in labeler.get_topic_labels(k, top_n=5)))
            for word, prob in mdl.get_topic_words(k, top_n=10):
                print(word, prob, sep='\t')
            print()

        return mdl

    def dmr_model(self, text_data, pair_map, save_path, topic_number=20):
        mdl = tp.DMRModel(tw=tp.TermWeight.ONE, min_cf=3, rm_top=5, k=topic_number)
        print(mdl.perplexity)

        index=0
        for doc in text_data:
            print(str(index) + " : " + str(doc))
            year=pair_map[index]
            mdl.add_doc(doc,metadata=year)
            index+=1

        mdl.burn_in = 100
        mdl.train(0)
        print('Num docs:', len(mdl.docs), ', Vocab size:', mdl.num_vocabs, ', Num words:', mdl.num_words)
        print('Removed top words:', mdl.removed_top_words)
        print('Training...', file=sys.stderr, flush=True)
        for i in range(0, 1000, 10):
            mdl.train(10)
            print('Iteration: {}\tLog-likelihood: {}'.format(i, mdl.ll_per_word))

        print('Saving...', file=sys.stderr, flush=True)
        mdl.save(save_path, True)

        # extract candidates for auto topic labeling
        extractor = tp.label.PMIExtractor(min_cf=20, min_df=5, max_len=5, max_cand=10000)
        cands = extractor.extract(mdl)

        # ranking the candidates of labels for a specific topic
        labeler = tp.label.FoRelevance(mdl, cands, min_df=5, smoothing=1e-2, mu=0.25)

        topic_contents = []
        for k in range(mdl.k):
            topic_ct = []
            print("== Topic #{} ==".format(k))
            print("Labels:", ', '.join(label for label, score in labeler.get_topic_labels(k, top_n=5)))
            for word, prob in mdl.get_topic_words(k, top_n=20):
                topic_ct.append((word, prob))
                print(word, prob, sep='\t')
            print()
            topic_contents.append(topic_ct)

        topic_contents_features = pd.DataFrame(topic_contents)
        topic_contents_features.to_csv('dmr_topic_feature.csv',index=False)
        # Init output
        topics_features = pd.DataFrame()
        col_features = []

        for k in range(mdl.k):
            print('Topic #{}'.format(k))
            array_features=[]
            features={}
            for m in range(mdl.f):
                #print('feature ' + mdl.metadata_dict[m] + " --> " + str(mdl.lambdas[k][m]) + " ")
                features[mdl.metadata_dict[m]]=mdl.lambdas[k][m]
                array_features.append(mdl.lambdas[k][m])
                if int(k) == 0:
                    #print('feature ' + mdl.metadata_dict[m])
                    col_features.append(mdl.metadata_dict[m])

            a = np.array(array_features)
            median=np.median(a)
            max=np.max(a)
            min=np.min(a)

            new_features=[]
            #new_features.append(k)
            for col in col_features:
                val = features[col]
                final_val = abs(max) + val + abs(median)
                features[col] = final_val
                #print("YYYYYY " + col + " :: " + str(features[col]))
                new_features.append(final_val)

            topics_features = topics_features.append(pd.Series(new_features), ignore_index=True)
            print("median " + str(median) + " : " + str(max) + " : " + str(min))

            for word, prob in mdl.get_topic_words(k):
                print('\t', word, prob, sep='\t')

        col_feaures = sorted(col_features, reverse=False)
        #col_features.insert(0,'Topic ID')
        topics_features.columns = col_features

        topics_features.to_csv('dmr_topic_year.csv', sep=',', encoding='utf-8')
        print(topics_features.head(20))

        df1_transposed = topics_features.T.rename_axis('Date').reset_index().sort_values(by='Date')
        print(f'type df1_t {type(df1_transposed)}')
        print(f'len df1_t {len(df1_transposed)}')
        print(f'shape of df1_t {df1_transposed.shape}')

        #labels = []
        #for i in range(0,topic_number-1):
        #    labels.append('Topic_'+str(i))
        #df1_transposed.columns=labels

        import seaborn as sns
        import matplotlib.colors as mcolors

        print(df1_transposed.head(20))
        df1_transposed = df1_transposed.melt('Date', var_name='Topic', value_name='Importance Score')
        g = sns.relplot(x="Date", y="Importance Score", hue='Topic', dashes=False, markers=True,  data=df1_transposed, kind='line')

        output = 'dmr_topic.png'
        g.fig.suptitle('DMR Topic Model Results')
        g.savefig(output, format='png', dpi=500)
        # Show the plot
        plt.show()

        # calculate topic distribution for each metadata using softmax
        probs = np.exp(mdl.lambdas - mdl.lambdas.max(axis=0))
        probs /= probs.sum(axis=0)

        print('Topic distributions for each metadata')
        for f, metadata_name in enumerate(mdl.metadata_dict):
            print(metadata_name, probs[:, f], '\n')

        x = np.arange(mdl.k)
        width = 1 / (mdl.f + 2)

        fig, ax = plt.subplots()
        for f, metadata_name in enumerate(mdl.metadata_dict):
            ax.bar(x + width * (f - mdl.f / 2), probs[:, f], width, label=mdl.metadata_dict[f])

        ax.set_ylabel('Probabilities')
        ax.set_yscale('log')
        ax.set_title('Topic distributions')
        ax.set_xticks(x)
        ax.set_xticklabels(['Topic #{}'.format(k) for k in range(mdl.k)])
        ax.legend()

        fig.tight_layout()
        plt.show()

        return mdl

