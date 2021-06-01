import pandas as pd
import csv
import os

file_name = '/Users/george/testData/seoul_data/dml_seoul_city_complaints_2020_2021.csv'

output_file = 'dml_seoul_city_complaints_2020_2021.csv'
# with open(file_name,'r') as file:
#     reader = csv.reader(file)
#     #print(next(reader))
#     result = []
#     for row in reader:
#         date = '.'.join(row[1].split('-')[:2])
#         complaints = ' '.join(row[2].split('\n')[1:])
#         answer = ' '.join(row[3].split('\n')[1:-3])
#         result.append([row[0], date, complaints, answer])
#         #print(complaints)
#         if len(complaints) < 10:
#             pass
# print(result)
# df = pd.DataFrame(result,columns=['idx','date','complaints','answers'])

#df.to_csv(output_file, index=False,encoding='utf-8')

file = '/Users/george/testData/seoul_data'
f = []

for i in os.listdir(file)[1:]:
    print(i)
    data_path = os.path.join(file, i)
    data = pd.read_csv(data_path)
    print(data.info())
    c= 0
    for idx, j in enumerate(data['answers']):
        if len(str(j)) < 10:
            print(data.loc[idx,:])
            c += 1
    print()
    print(c)
    f.append(data)

df = pd.concat(f, axis=0)
print(df.info())
print(df['answers'][8634:])

#df.to_csv('seoul_city_complaints_total.csv', index=False, encoding='utf-8')

