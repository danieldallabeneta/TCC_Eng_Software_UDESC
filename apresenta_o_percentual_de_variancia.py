from numpy import loadtxt
from numpy import unique

path = '' # informe o local do dataset
coluns = [] # informe as colunas a serem consideradas ex.: [0,1,2,3,5,10]
#leitura do dataset
data = loadtxt(path, delimiter=',', usecols=coluns)

for i in range(data.shape[1]):
 num = len(unique(data[:, i]))
 percentage = float(num) / data.shape[0] * 100
 print('%d, %d, %.2f%%' % (i, num, percentage))