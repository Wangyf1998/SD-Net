import pandas as pd

df = pd.read_csv("/home/wyf/mydoc/Artemis/artemis_new_withID.csv")
df['split'] = df['split'].apply(lambda x : 'train' if x == 'rest' else x)
df.to_csv('test.csv')


