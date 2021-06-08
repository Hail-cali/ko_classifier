import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'AppleGothic'
plt.style.use('seaborn')

df = pd.read_csv('dmr_topic_year.csv')
df = df.drop('Unnamed: 0', axis=1)
df_topic = pd.read_csv('dmr_topic_feature.csv')
df_topic = df_topic.drop('Unnamed: 0', axis=1)

df_t = df.T.rename_axis('Date').reset_index().sort_values(by='Date')
df_t = df_t.melt('Date', var_name='Topic', value_name='Importance Score')
df_topic_t = df_topic.T

for k in range(10):
    is_topic = (df_t.Topic==k)
    print(df_topic_t.iloc[:, k])
    k_topic_data = df_topic.iloc[:4, k]
    tokens = [token.split(',')[0][2:] for token in k_topic_data]
    fig, ax = plt.subplots((1,9))

    # g = sns.relplot(x="Date", y="Importance Score", hue='Topic', dashes=False, markers=True,  data=df_t[is_topic], kind='line')
    # g.fig.suptitle(f'DMR {k} Topic Model Results')
    #
    # g.fig.legend(tokens)
    #
    # output = f'../img/dmr_{k}_topic.png'
    #g.savefig(output, format='png', dpi=500)

plt.show()




