import pandas as pd
import re
import os

data = []
file_path = './seoul_data'

file_list = os.listdir(file_path)
print(file_list)
for file in file_list:
    path = os.path.join(file_path, file)
    docs = pd.read_csv(path)
    data.append(docs)

df = pd.concat(data, axis=0)
print(df.info())
del(data)

#df.to_csv('seoul_city_complaints_2019_2021.csv', index=False, encoding='utf-8')
