import csv
import itertools
import os

from sklearn.metrics import (f1_score, roc_auc_score, confusion_matrix, accuracy_score)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import tensorflow as tf
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from scikeras.wrappers import KerasClassifier
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from imblearn.over_sampling import ADASYN


def get_scores(y_test, y_pred, dataset, algorithm, rs, model, ws):
    scores = []
    scores.append(dataset)
    scores.append(algorithm)
    scores.append(ws)
    scores.append(model)
    scores.append(rs)

    scores.append(f1_score(y_test, y_pred, average='micro'))
    print("F1-Score(micro): " + str(scores[-1]))
    scores.append(f1_score(y_test, y_pred, average='macro'))
    print("F1-Score(macro): " + str(scores[-1]))
    scores.append(f1_score(y_test, y_pred, average='weighted'))
    print("F1-Score(weighted): " + str(scores[-1]))
    scores.append(f1_score(y_test, y_pred, average=None))
    print("F1-Score(None): " + str(scores[-1]))

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

    scores.append(accuracy_score(y_test, y_pred, normalize=True))
    print("Accuracy: " + str(scores[-1]))

    precision = tp/ (tp + fp)
    scores.append(precision)

    recall = tp / (tp + fn)
    scores.append(recall)
    print("Recall: " + str(scores[-1]))

    specificity = tn / (tn + fp)
    scores.append(specificity)
    print("Specificity: " + str(scores[-1]))

    cnf_matrix = confusion_matrix(y_test, y_pred)
    cnf_matrix = cnf_matrix.astype('float') / cnf_matrix.sum(axis=1)[:, np.newaxis]
    print("Confusion Matrix: [" + str(cnf_matrix[0][0]) + ", " + str(round(cnf_matrix[1][1], 2)) + "]")
    plot_confusion_matrix(cnf_matrix, dataset, algorithm)

    scores.append(roc_auc_score(y_test, y_pred))
    print("ROC AUC score: " + str(scores[-1]))

    scores.append([tn, fp, fn, tp])
    head = ['Dataset', 'Algoritm', 'window', 'model', 'resample', 'F1-Score(micro)', 'F1-Score(macro)',
            'F1-Score(weighted)', 'F1-Score(None)', 'Accuracy', 'precision', 'Recall', 'Specificity', 'ROC AUC score',
            'Confusion matrix']

    if not os.path.exists('results/' + dataset + '.csv'):
        f = open("results/" + dataset + ".csv", "a")
        writer = csv.writer(f)
        writer.writerow(head)
        f.close()

    f = open("results/" + dataset + ".csv", "a")
    writer = csv.writer(f)
    writer.writerow(scores)
    f.close()

    return scores

def plot_confusion_matrix(cm, dataset, algorithm, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()    

    fmt = '.2f' if normalize else 'f'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig('results/cf-' + dataset + '-' + algorithm +'.png')
    plt.close()

def RandomForest_(Xtrain, Ytrain, Xtest, Ytest, dataset, rs, model, ws):
    print("RANDOM FOREST")
    cv_score = []
    i = 1
    
    print("Setting stratified k-fold...")
    k = 10
    kf = StratifiedKFold(n_splits=k, shuffle=False)
    print("k =", k)
    print("... DONE!\n")

    print("TRAIN AND VALIDATION SETS:")
    for train_index, test_index in kf.split(Xtrain, Ytrain):
        print('{} of KFold {}'.format(i, kf.n_splits))
        xtr_RF, xvl_RF = Xtrain.iloc[train_index], Xtrain.iloc[test_index]
        ytr_RF, yvl_RF = Ytrain.iloc[train_index], Ytrain.iloc[test_index]

        rf = RandomForestClassifier(random_state=42, class_weight='balanced', n_estimators=100, n_jobs=-1)
        rf.fit(xtr_RF, ytr_RF.values.ravel())
        score = roc_auc_score(yvl_RF, rf.predict(xvl_RF))
        print('ROC AUC score:', score)
        cv_score.append(score)
        i += 1

    print('\nCROSS VALIDANTION SUMMARY:')
    print('Mean: ' + str(np.mean(cv_score)))
    print('Std deviation: ' + str(np.std(cv_score)))

    print("\nTEST SET:")
    get_scores(Ytest, rf.predict(Xtest), dataset, "RandomForest", rs, model, ws)

def read_metric_columns(file):
    with open(file + '.txt', 'r') as arquivo:
        conteudo = arquivo.read()
    dados = [valor.strip() for valor in conteudo.split(',')]
    return dados

if __name__ == '__main__':
     
    main_columns = read_metric_columns('all_columns_exec')
    structural_metrics = read_metric_columns('metricas_estruturais_exec')         
    evolutionary_metrics = read_metric_columns('metricas_evolutivas_exec')
    technical_debt_metrics = read_metric_columns('metricas_divida_tecnica_exec')
    change_distiller_metrics = ['change']

    model1 = structural_metrics + evolutionary_metrics
    model2 = model1 + technical_debt_metrics
    
    dataset_list = []#'projectA','projectB'
    datasets = [] #'projectA'   

    resamples = ['ADA']#'NONE', 'ROS', 'SMOTE', 'ADA'

    windowsize = [0]
    models = [{'key': 'model1', 'value': model1}, {'key': 'model2', 'value': model2}]

    for dataset in datasets:
        for ws in windowsize:
            for rs in resamples:
                for model in models:
                    if dataset == 'all':
                        dfs = []
                        for ds_name in dataset_list:
                            dfs.append(pd.read_csv('../6.join_metrics/results/' + ds_name + '-all-releases.csv'))
                        all_releases_df = pd.concat(dfs)
                    else:                        
                        all_releases_df = pd.read_csv(dataset + '.csv')
                    
                    all_releases_df = all_releases_df.fillna(0)
                    all_releases_df.columns = main_columns                    
                    print("Filtering required columns into X features...")
                    X = all_releases_df[model.get('value')].copy()
                    print("... DONE!")
                    
                    print("Setting y column containing label of change-proneness...")
                    y = pd.DataFrame(all_releases_df.loc[:, 'change'])
                    print("... DONE!")
                    print("Declaring a dictionary to save results...")
                    results_dict = dict()
                    print("... DONE!")
                    print("Splitting dataset into train and test sets...")
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)
                    print("General information:")
                    print("X Train set:", X_train.shape[0], "X Test set:", X_test.shape[0])
                    print("y Train set:", y_train.shape[0], "y Test set:", y_test.shape[0])
                    print("... DONE!")

                    print("Scaling features...")
                    scaler = MinMaxScaler()
                    X_train = pd.DataFrame(scaler.fit_transform(X_train))
                    X_test = pd.DataFrame(scaler.fit_transform(X_test))
                    print("... DONE!")

                    y_test = pd.DataFrame(y_test)
                    y_train = pd.DataFrame(y_train)

                    if rs == 'ADA':                       
                        ada = ADASYN(random_state=42)
                        X_ADA, y_ADA = ada.fit_resample(X_train, y_train)
                        RandomForest_(X_ADA, y_ADA, X_test, y_test, dataset, rs, model.get('key'), ws)  