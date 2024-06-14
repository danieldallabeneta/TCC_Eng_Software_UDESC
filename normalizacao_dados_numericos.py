from sklearn import preprocessing
import numpy as np
from pandas import read_csv
import pandas as pd

path = '' # informe o caminho do dataset
colunas = [] # informe as colunas a serem consideradas. Ex.: ['colunaA', 'colunaB']

# new DataFrame para armazenar os dados normalizados.
dataFrame = pd.DataFrame()

# leitura do dataset
df = read_csv(path, usecols=colunas)

for c in colunas:
    x_array = np.array(df[c])
    normalized_arr = preprocessing.normalize([x_array])    
    dados_inserir = [{c:normalized_arr[0]}]
    for dados in dados_inserir:
        coluna, valores = next(iter(dados.items()))
        dataFrame[coluna] = valores

#print(dataFrame)
# gera um csv com os dados normalizados.
dataFrame.to_csv('normalized.csv',index=True)