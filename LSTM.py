# Stacked LSTM for international airline passengers problem with memory
import numpy as np
import matplotlib.pyplot as plt
from pandas import read_csv
import pandas as pd
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM, Dropout
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

def create_dataset(dataset, look_back=1):
	dataX, dataY = [], []
	for i in range(len(dataset)-look_back-1):
		a = dataset[i:(i+look_back), 0]
		dataX.append(a)
		dataY.append(dataset[i + look_back, 0])
	return np.array(dataX), np.array(dataY)

if __name__ == '__main__':

    main_columns = ['project','release','hash','file','previouscommit','classcurrentcommit','classpreviouscommit',
                    'change','cbo','cbomodified','fanin','fanout','wmc','dit','noc','rfc','lcom','lcom2','tcc','lcc',
                    'totalmethodsqty','staticmethodsqty','publicmethodsqty','privatemethodsqty','protectedmethodsqty',
                    'defaultmethodsqty','visiblemethodsqty','finalmethodsqty','synchronizedmethodsqty','totalfieldsqty',
                    'staticfieldsqty','publicfieldsqty','privatefieldsqty','protectedfieldsqty','finalfieldsqty','nosi',
                    'loc','returnqty','loopqty','comparisonsqty','trycatchqty','parenthesizedexpsqty','stringliteralsqty',
                    'numbersqty','assignmentsqty','mathoperationsqty','variablesqty','maxnestedblocksqty','anonymousclassesqty',
                    'innerclassesqty','uniquewordsqty','avgcyclomatic','countclassbase','countclasscoupled','countclassderived',
                    'countdeclclassmethod','countdeclclassvariable','countdeclinstancemethod','countdeclinstancevariable','countdeclmethod',
                    'countdeclmethodall','countdeclmethoddefault','countdeclmethodprivate','countdeclmethodprotected','countdeclmethodpublic',
                    'countline','countlineblank','countlinecode','countlinecodedecl','countlinecodeexe','countlinecomment','countsemicolon',
                    'countstmt','countstmtdecl','countstmtexe','maxcyclomatic','maxinheritancetree','percentlackofcohesion','ratiocommenttocode',
                    'sumcyclomatic','maxnesting','lch','creationdate','closedate','creationcommithash','closecommithash','type','squid','severity',
                    'startline','endline','resolution','status','effort','debt','author','hashclose','releaseclose']
	
    structural_metrics = ['cbo','cbomodified','fanin','fanout','wmc','dit','noc','rfc','lcom','lcom2','tcc','lcc',
                          'totalmethodsqty','staticmethodsqty','publicmethodsqty','privatemethodsqty','protectedmethodsqty',
                          'defaultmethodsqty','visiblemethodsqty','finalmethodsqty','synchronizedmethodsqty','totalfieldsqty',
                          'staticfieldsqty','publicfieldsqty','privatefieldsqty','protectedfieldsqty','finalfieldsqty',
                          'nosi','loc','returnqty','loopqty','comparisonsqty','trycatchqty','parenthesizedexpsqty','stringliteralsqty',
                          'numbersqty','assignmentsqty','mathoperationsqty','variablesqty','maxnestedblocksqty','anonymousclassesqty',
                          'innerclassesqty','uniquewordsqty','avgcyclomatic','countclassbase','countclasscoupled','countclassderived',
                          'countdeclclassmethod','countdeclclassvariable','countdeclinstancemethod','countdeclinstancevariable',
                          'countdeclmethod','countdeclmethodall','countdeclmethoddefault','countdeclmethodprivate','countdeclmethodprotected',
                          'countdeclmethodpublic','countline','countlineblank','countlinecode','countlinecodedecl','countlinecodeexe',
                          'countlinecomment','countsemicolon','countstmt','countstmtdecl','countstmtexe','maxcyclomatic','maxinheritancetree',
                          'percentlackofcohesion','ratiocommenttocode','sumcyclomatic','maxnesting']
    
    evolutionary_metrics = ['lch']

    technical_debt_metrics = ['type','severity','resolution','status','effort']

    change_distiller_metrics = ['change']

    model1 = structural_metrics + evolutionary_metrics
    model2 = model1 + technical_debt_metrics

    models = [{'key': 'model1', 'value': model1}, {'key': 'model2', 'value': model2}]

    datasets = ['exec']
   
    tf.random.set_seed(7)

    for data in datasets:
        for model in models:
            
            all_releases_df = pd.read_csv(data + '.csv', usecols=model.get('value'))
            all_releases_df = all_releases_df.fillna(0)
            dataset = all_releases_df.values
            dataset = dataset.astype('float32')
            scaler = MinMaxScaler(feature_range=(0, 1))
           
            train_size = int(len(dataset) * 0.7)
            test_size = len(dataset) - train_size
            train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]
            
            look_back = 3
            trainX, trainY = create_dataset(train, look_back)
            testX, testY = create_dataset(test, look_back)
            
            trainX = np.reshape(trainX, (trainX.shape[0], trainX.shape[1], 1))
            testX = np.reshape(testX, (testX.shape[0], testX.shape[1], 1))
            
            batch_size = 10
            model = Sequential()
           
            model.add(LSTM(50, return_sequences=True))
            model.add(Dropout(0.2))
            
            model.add(LSTM(50))
            model.add(Dropout(0.2))
            model.add(Dense(1, activation='sigmoid'))
            model.compile(loss='mean_squared_error', optimizer='adam',metrics=['accuracy'])
            for i in range(100):
                model.fit(trainX, trainY, epochs=1, batch_size=batch_size, verbose=2, shuffle=False)
                for layer in model.layers:
                    if hasattr(layer, 'reset_states'):
                        layer.reset_states()
                            
            trainPredict = model.predict(trainX, batch_size=batch_size)
            for layer in model.layers:
                if hasattr(layer, 'reset_states'):
                    layer.reset_states()
            
            testPredict = model.predict(testX, batch_size=batch_size)
           
            scaler.fit(trainPredict)
            trainPredict = scaler.inverse_transform(trainPredict)
            trainY = scaler.inverse_transform([trainY])
            testPredict = scaler.inverse_transform(testPredict)
            testY = scaler.inverse_transform([testY])
            
            trainScore = np.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
            print('Train Score: %.2f RMSE' % (trainScore))
            testScore = np.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
            print('Test Score: %.2f RMSE' % (testScore))

            y_pred_binary = (testY > 0.5).astype(int)
            accuracy = accuracy_score(testY, y_pred_binary)
            print("Test Accuracy:", accuracy)
           
            trainPredictPlot = np.empty_like(dataset)
            trainPredictPlot[:, :] = np.nan
            trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict
            
            testPredictPlot = np.empty_like(dataset)
            testPredictPlot[:, :] = np.nan
            testPredictPlot[len(trainPredict)+(look_back*2)+1:len(dataset)-1, :] = testPredict
            
            plt.plot(scaler.inverse_transform(dataset))
            plt.plot(trainPredictPlot)
            plt.plot(testPredictPlot)
            plt.show()
