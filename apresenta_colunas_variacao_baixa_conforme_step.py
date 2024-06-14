# explore the effect of the variance thresholds on the number of selected features
from numpy import arange
from pandas import read_csv
from sklearn.feature_selection import VarianceThreshold
from matplotlib import pyplot

path = '' # informe o local do arquivo
colunas = [] # informe as colunas a serem consideradas

# utilize a leitura do csv abaixo se considerar todas as colunas
#df = read_csv(path, header=None)

# utilize a leitura do csv abaixo para considerar apenas as colunas informadas
df = read_csv(path, usecols=colunas)

# split data into inputs and outputs
data = df.values
X = data[:, :-1]
y = data[:, -1]
print(X.shape, y.shape)
# define thresholds to check
thresholds = arange(0.0, 0.55, 0.05)

# apply transform with each threshold
results = list()
for t in thresholds:
	# define the transform
	transform = VarianceThreshold(threshold=t)
	# transform the input data
	X_sel = transform.fit_transform(X)
	# determine the number of input features
	n_features = X_sel.shape[1]
	print('>Threshold=%.2f, Features=%d' % (t, n_features))
	# store the result
	results.append(n_features)

	indices_selecionados = transform.get_support(indices=True)
	indices_removidos = [i for i in range(len(colunas)) if i not in indices_selecionados]
	colunas_removidas = [colunas[i] for i in indices_removidos]

	for coluna in colunas_removidas:
		print(coluna)

# imprime gráfico relacionando a quantidade de features a serem mantidas de acordo com o thresholds aplicado.
pyplot.plot(thresholds, results)
pyplot.show()