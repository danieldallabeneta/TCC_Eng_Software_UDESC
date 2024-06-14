from pandas import read_csv
from datetime import datetime
from matplotlib import pyplot

# load data
def parse(x):
	return datetime.strptime(x, '%Y %m %d %H')

dataset = read_csv('exec.csv',  header=0, index_col=0)

#dataset.drop('No', axis=1, inplace=True)
# manually specify column names
dataset.columns = ['cbo','cbomodified','fanin','fanout','wmc','dit','noc','rfc','lcom','lcom2','tcc','lcc','totalmethodsqty','staticmethodsqty','publicmethodsqty','privatemethodsqty','protectedmethodsqty','defaultmethodsqty','visiblemethodsqty','finalmethodsqty','synchronizedmethodsqty','totalfieldsqty','staticfieldsqty','publicfieldsqty','privatefieldsqty','protectedfieldsqty','finalfieldsqty','nosi','loc','returnqty','loopqty','comparisonsqty','trycatchqty','parenthesizedexpsqty','stringliteralsqty','numbersqty','assignmentsqty','mathoperationsqty','variablesqty','maxnestedblocksqty','anonymousclassesqty','innerclassesqty','uniquewordsqty','avgcyclomatic','countclassbase','countclasscoupled','countclassderived','countdeclclassmethod','countdeclclassvariable','countdeclinstancemethod','countdeclinstancevariable','countdeclmethod','countdeclmethodall','countdeclmethoddefault','countdeclmethodprivate','countdeclmethodprotected','countdeclmethodpublic','countline','countlineblank','countlinecode','countlinecodedecl','countlinecodeexe','countlinecomment','countsemicolon','countstmt','countstmtdecl','countstmtexe','maxcyclomatic','maxinheritancetree','percentlackofcohesion','ratiocommenttocode','sumcyclomatic','maxnesting','lch','type','severity','resolution','status','effort']
dataset.index.name = 'change'
# mark all NA values with 0
dataset.fillna(0, inplace=True)
# summarize first 5 rows
print(dataset.head(5))
# save to file
dataset.to_csv('exec_d.csv')

dataset = read_csv('exec_d.csv', header=0, index_col=0)
values = dataset.values
# specify columns to plot
groups = [0, 1, 2, 3, 5, 6, 7]
i = 1
# plot each column
pyplot.figure()
for group in groups:
 pyplot.subplot(len(groups), 1, i)
 pyplot.plot(values[:, group])
 pyplot.title(dataset.columns[group], y=0.5, loc='right')
 i += 1
pyplot.show()