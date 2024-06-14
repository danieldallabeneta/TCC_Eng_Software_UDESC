from pandas import read_csv

path = '' # informe o local do seu arquivo
coluns = [] # informe as colunas que devem ser avaliadas.

# load the dataset
df = read_csv(path)

# summarize the number of unique values in each column
dados = df.nunique()

# gera um arquivo csv com as colunas e a quantidade de valores atribuidos a cada uma.
dados.to_csv('results.csv', index=True)