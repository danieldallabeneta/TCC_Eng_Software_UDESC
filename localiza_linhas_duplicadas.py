from pandas import read_csv

path = '' # informe o caminho do dataset
colunas = [] # informe as coluna a serem consideradas para avaliação Ex.: ['colunaA', 'colunaB']
# load the dataset
# use a opção de leitura abaixo para considerar apenas as colunas informadas
#df = read_csv(path, usecols=colunas)

# use a opção de leitura abaixo para considerar todas as colunas do dataset
df = read_csv(path, header=None)

# calculate duplicates
dups = df.duplicated()
dups.to_csv('duplicated.csv',index=True);
# report if there are any duplicates
data = dups.any()
print(data)
df[dups].to_csv('results.csv',index=True);
# list all duplicate rows
print(df[dups])