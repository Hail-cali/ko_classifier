import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'AppleGothic'

df = pd.read_csv('dmr_topic_year.csv')
df = df.drop('Unnamed: 0', axis=1)

df_t = df.T.rename_axis('Date').reset_index().sort_values(by='Date')

df_t = df_t.melt('Date', var_name='Topic', value_name='Importance Score')

df_topic = pd.read_csv('dmr_topic_feature.csv')
df_topic_T = df_topic.T.rename_axis('token').reset_index()

for k in range(10):
    is_topic = (df_t.Topic==k)
    print(df_topic_T.loc[:10, k])
    k_topic_data = df_topic_T.loc[:4, k]
    tokens = [token.split(',')[0][2:] for token in k_topic_data]
    g = sns.relplot(x="Date", y="Importance Score", hue='Topic', dashes=False, markers=True,  data=df_t[is_topic], kind='line')
    g.fig.suptitle(f'DMR {k} Topic Model Results')
    g.fig.legend(tokens[:2])

    output = f'../img/dmr_{k}_topic.png'
    #g.savefig(output, format='png', dpi=500)

for k in range(10):
    k_topic_data = df_topic_T.loc[:10, k]
    tokens = [token.split(',')[0][2:] for token in k_topic_data]
plt.show()




