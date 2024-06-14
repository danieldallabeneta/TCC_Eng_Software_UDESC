import pandas as pd
from pandas import read_csv
import matplotlib.pyplot as plt
import numpy as np

path = '' #Informe o Path
colunas = [] #Informe as coluns que devem ser considerar para efetuar o calculo

df = read_csv(path, usecols=colunas)
coef_corr = df.corr()
correlacoes = pd.DataFrame(coef_corr)

#Cria um arquivo csv contendo os valores calculados de correlação    
correlacoes.to_csv('correlacao.csv', index=True)

#Criar gráfico de bolhas para apresentação dos valores calculados.

# Extrair as coordenadas x, y e tamanhos das bolhas
x = []
y = []
sizes = []
colors = []

min_size = 10
scale_factor = 200

for i, col in enumerate(correlacoes.columns):
    for j, row in enumerate(correlacoes.index):
        x.append(i)
        y.append(j)
        sizes.append(abs(correlacoes.iloc[i, j]) * scale_factor + min_size)
        colors.append(correlacoes.iloc[i, j])

# Criar o gráfico de bolhas
plt.figure(figsize=(15, 15))
sc = plt.scatter(x, y, s=sizes, c=colors, cmap='viridis', alpha=0.7, vmin=-1, vmax=1)

# Adicionar etiquetas de eixo e título
plt.xticks(range(len(correlacoes.columns)), correlacoes.columns, rotation=90, fontsize=8)  # Ajuste o tamanho do texto
plt.yticks(range(len(correlacoes.columns)), correlacoes.columns, fontsize=8)  # Ajuste o tamanho do texto
plt.title('Matriz de Correlação de Pearson')
plt.xlabel('Métricas')
plt.ylabel('Métricas')

cbar = plt.colorbar(sc)
cbar.set_label('Correlação de Pearson')
cbar.set_ticks(np.linspace(-1, 1, 11))

# Exibir o gráfico
plt.tight_layout()
# apresentar o gráfico gerado
plt.show()
# criar um arquivo em pdf com o gráfico gerado;
plt.savefig('correlacao.pdf')