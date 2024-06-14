from pandas import read_csv

path = '' # informe o caminho do dataset

# leitura do csv
df = read_csv(path, header=None)
print(df.shape)

# get number of unique values for each column
counts = df.nunique()
# record columns to delete
to_del = [i for i,v in enumerate(counts) if v <= 1]
print(to_del)
# drop useless columns
df.drop(to_del, axis=1, inplace=True)
# export to csv new Data Frame
#df.to_csv('results.csv', index=True)
print(df.shape)