from pandas import read_csv
from pprint import pprint
import numpy as np
import pandas as pd
from scipy import stats
from matplotlib import pyplot
from pandas.plotting import scatter_matrix
from os import path, getcwd, listdir
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD
from sklearn.svm import SVC
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.gaussian_process.kernels import RBF
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LinearRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import ComplementNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import CategoricalNB
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
from pre_processing.base import BaseReduction
from sklearn.metrics import plot_confusion_matrix
from sklearn import tree
from sklearn.preprocessing import normalize

class Supervisioned:

    def __init__(self, scenario):
        self.path = path.abspath(getcwd())
        self.datasetPath = path.join(self.path, 'data', 'modified', scenario)
        self.BaseReduzida1 = None
        self.BaseReduzida2 = None
        self.BaseReduzida3 = None
        self.BaseOriginal = None
        self.bases = None
        #self.bases = ['BaseOriginal', 'BaseReduzida1', 'BaseReduzida2', 'BaseReduzida3']



    def generateBases(self):
        baseReduction = BaseReduction()

        names = ['Speed', 'Operatorname', 'CellID', 
				'RSRP', 'RSRQ', 'SNR', 'RSSI', 'State', 
				'ServingCell_Distance', 'CQI']

        dataset = read_csv(self.datasetPath, usecols=names)
        

        print('Conversão de categorias em números')
        label_encoder_X = LabelEncoder()
        dataset['Operatorname'] = label_encoder_X.fit_transform(dataset['Operatorname'])
        dataset['State'] = label_encoder_X.fit_transform(dataset['State'])

        print('--- Gerando BaseOriginal')
        self.BaseOriginal = baseReduction.original(dataset)
        print('--- BaseOriginal:', dataset.shape)

        print('--- Gerando BaseReduzida1')
        self.BaseReduzida1 = baseReduction.reduction1(dataset)
        print('--- BaseReduzida1:', self.BaseReduzida1.shape)
        
        print('--- Gerando BaseReduzida2')
        self.BaseReduzida2 = baseReduction.reduction2(dataset)
        print('--- BaseReduzida2:', self.BaseReduzida2.shape)

        print('--- Gerando BaseReduzida3')
        self.BaseReduzida3 = baseReduction.reduction3(dataset)
        print('--- BaseReduzida3:', self.BaseReduzida3.shape)
        
        self.bases = { 'BaseOriginal': self.BaseOriginal,
            'BaseReduzida1': self.BaseReduzida1,
            'BaseReduzida2': self.BaseReduzida2,
            'BaseReduzida3': self.BaseReduzida3
        }

        #self.kNN()
        self.ad()
        #self.nb()
        #self.mlp()



    def mlp(self):
        print('----------------------------------------------------------')
        print('Início Experimento Redes Neurais ')

        '''
        skf = StratifiedKFold(n_splits=2)

        mlps = []
        momentum = 0.8
        learningRate = [0.1, 0.01, 0.001]
        maxIter = [500, 2000, 10000]
        hiddenLayers = [4, 8, 12]
        i = 1
        for mi in maxIter:
            for hl in hiddenLayers:
                for lr in learningRate:
                    print(i, mi, hl, lr)
                    mlps.append((f'{i}', MLPClassifier(random_state=0, solver='sgd', momentum=momentum, max_iter=mi, learning_rate='adaptive', learning_rate_init=lr, hidden_layer_sizes=hl)))
                    i += 1
        '''
        results = {}

        for base in self.bases:
            print('----------------')
            print(base, 'cross_val_score treinamento')
            print('---------------')
            dataset = self.bases[base]
            feature_names = ['Speed', 'Operatorname', 'CellID', 
				'RSRP', 'RSRQ', 'SNR', 'RSSI', 'State', 
				'ServingCell_Distance', 'CQI']
            class_names = ['high', 'low', 'medium']
            array = dataset.values
            arrLen = len(dataset.columns) - 1
            X = array[:,0:arrLen]
            y = array[:,arrLen]
            results[base] = {}
            X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.20, random_state=50)
            score = []
            score2 = []
            names = []



            '''
            for name, model in mlps:

                cv_results = cross_val_score(model, X_train, Y_train, cv=skf, scoring='accuracy')
                results[base][name] = {}
                results[base][name]['mean'] = cv_results.mean()
                results[base][name]['std'] = cv_results.std()
                print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))
                mean = results[base][name]['mean']
                score.append(mean)
                names.append(name)
                score2.append(cv_results)

            sortArr = sorted(score, reverse=True)
            fig = pyplot.figure()
            ax = fig.add_subplot(111)
            bp = ax.boxplot(score2, labels=names)
            ax.set_title(f'{base}') 
            fig.savefig(f'figs/mlp/{base}-mlp.png')
            bestValue = sortArr[0]


            for name, model in mlps:
                mean = results[base][name]['mean']
                if mean == bestValue:
                    print(f'{base} com score médio de {bestValue}. Melhor parâmetro (index): {name}')
                    model.set_params()
                    clf = model.fit(X_train, Y_train)
                    predictions = model.predict(X_test)
                    
                    # Evaluate predictions
                    print('Acurácia de predição:', accuracy_score(Y_test, predictions))
                    print('Matriz de confusão:')
                    print(confusion_matrix(Y_test, predictions))
                    print('Report de classificação:')
                    print(classification_report(Y_test, predictions))                 
            '''

            skf = StratifiedKFold(n_splits=10)
            selectedMlps = []
            mapp = [0.00001, 0.001, 0.01, 0.1, 0.2]
            mlp = 1
            for i in mapp:
                selectedMlps.append((f'MLP-{mlp}', MLPClassifier(alpha=i, momentum=0.8, max_iter=500, learning_rate_init=0.1, hidden_layer_sizes=12)))
                mlp += 1


            for name, model in selectedMlps:
                cv_results = cross_val_score(model, X_train, Y_train, cv=skf, scoring='accuracy')
                results[base][name] = {}
                results[base][name]['mean'] = cv_results.mean()
                results[base][name]['std'] = cv_results.std()
                print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))
                mean = results[base][name]['mean']
                score.append(mean)
                names.append(name)
                score2.append(cv_results)

            sortArr = sorted(score, reverse=True)
            #fig = pyplot.figure()
            #ax = fig.add_subplot(111)
            #bp = ax.boxplot(score2, labels=names)
            #ax.set_title(f'{base}') 
            #fig.savefig(f'figs/mlp/{base}-mlp.png')
            bestValue = sortArr[0]
            
            acuracias = []
            for name, model in selectedMlps:
                #mean = results[base][name]['mean']
                #if mean == bestValue:
                #print(f'{base} com score médio de {bestValue}. Melhor parâmetro (index): {name}')
                #model.set_params()
                clf = model.fit(X_train, Y_train)
                predictions = model.predict(X_test)
                
                # Evaluate predictions
                print(f'{name} Acurácia de predição:', accuracy_score(Y_test, predictions))
                a = accuracy_score(Y_test, predictions)
                acuracias.append(a)
                #print('Matriz de confusão:')
                #print(confusion_matrix(Y_test, predictions))
                #print('Report de classificação:')
                #print(classification_report(Y_test, predictions))
            mean = np.mean(acuracias)
            std = np.std(acuracias)
            print(f'Acurácia Média: {mean} ({std})')

            table2_mean = []
            for i in range(1,4):
                for name, model in selectedMlps:
                    size = i * 10
                    randomState = i * 15
                    X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=size, random_state=randomState)
                    cv_results = cross_val_score(model, X_train, Y_train, cv=skf, scoring='accuracy')
                    print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))
                    mean = cv_results.mean()
                    table2_mean.append(mean)

            mean = np.mean(table2_mean)
            std = np.std(table2_mean)
            print(f'3x10-Fold Cross Validation: {mean} ({std})')
            



    def nb(self):
        print('----------------------------------------------------------')
        print('Início Experimento Naive Bayes ')
        
        skf = StratifiedKFold(n_splits=2)
        nbParameters = []
        nbParameters.append((f'NB', GaussianNB()))
        #nbParameters.append((f'NB2', MultinomialNB()))
        #nbParameters.append((f'NB3', ComplementNB()))
        #nbParameters.append((f'NB4', BernoulliNB()))
        #nbParameters.append((f'NB5', CategoricalNB()))

        results = {}
        mediaGeral = []
        for base in self.bases:
            print('----------------')
            print(base, 'cross_val_score treinamento')
            print('---------------')
            dataset = self.bases[base]
            #dataset.hist()
            #pyplot.show()
            feature_names = ['Speed', 'Operatorname', 'CellID', 
				'RSRP', 'RSRQ', 'SNR', 'RSSI', 'State', 
				'ServingCell_Distance', 'CQI']
            class_names = ['high', 'low', 'medium']
            array = dataset.values
            arrLen = len(dataset.columns) - 1
            X = array[:,0:arrLen]
            y = array[:,arrLen]
            results[base] = {}
            X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.20, random_state=50)
            score = []
            score2 = []
            names = []

            for name, model in nbParameters:

                cv_results = cross_val_score(model, X_train, Y_train, cv=skf, scoring='accuracy')
                results[base][name] = {}
                results[base][name]['mean'] = cv_results.mean()
                results[base][name]['std'] = cv_results.std()
                print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))
                mean = results[base][name]['mean']
                score.append(mean)
                names.append(name)
                score2.append(cv_results)

            sortArr = sorted(score, reverse=True)
            fig = pyplot.figure()
            ax = fig.add_subplot(111)
            bp = ax.boxplot(score2, labels=names)
            ax.set_title(f'{base}') 
            fig.savefig(f'figs/nb/{base}-nb.png')
            bestValue = sortArr[0]
            acuracias = []
            for name, model in nbParameters:
                #mean = results[base][name]['mean']
                #if mean == bestValue:
                print(f'{name} {base} com score médio de {bestValue}.')
                clf = model.fit(X_train, Y_train)
                predictions = model.predict(X_test)
                t = model.predict_proba(X_test)
                print()
                #print(dir(model))
                # Evaluate predictions
                print('Acurácia de predição:', accuracy_score(Y_test, predictions))
                a = accuracy_score(Y_test, predictions)
                acuracias.append(a)
                #print('Matriz de confusão:')
                #print(confusion_matrix(Y_test, predictions))
                #print('Report de classificação:')
                #print(classification_report(Y_test, predictions))#, zero_division=1))
            mean = np.mean(acuracias)
            std = np.std(acuracias)
            print(f'Acurácia Média: {mean} ({std})')
            
            table2_mean = []
            for i in range(1,4):
                for name, model in nbParameters:
                    size = i * 10
                    randomState = i * 15
                    X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=size, random_state=randomState)
                    cv_results = cross_val_score(model, X_train, Y_train, cv=skf, scoring='accuracy')
                    print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))
                    mean = cv_results.mean()
                    table2_mean.append(mean)

            mean = np.mean(table2_mean)
            std = np.std(table2_mean)
            mediaGeral.append(mean)
            print(f'3x10-Fold Cross Validation: {mean} ({std})')
        
        mediaGeral_result = np.mean(mediaGeral)
        print(f'MediaGeral: {mediaGeral_result}')


    def ad(self):
        print('----------------------------------------------------------')
        print('Início Experimento AD - 10-fold cross validation. P-> Com Poda')
        skf = StratifiedKFold(n_splits=10)
        
        adParameters = []
        pruning = [0.1, 0.01, 0.001, 0.0001, 0.00001]
        adParameters.append((f'AD', DecisionTreeClassifier()))
        for i in pruning:
            adParameters.append((f'AD-P', DecisionTreeClassifier(ccp_alpha=i)))
        #adParameters.append((f'AD-Poda2', DecisionTreeClassifier(ccp_alpha=0.2)))

        results = {}

        for base in self.bases:
            print('----------------')
            print(base, 'cross_val_score treinamento')
            print('---------------')
            dataset = self.bases[base]
            feature_names = ['Speed', 'Operatorname', 'CellID', 
				'RSRP', 'RSRQ', 'SNR', 'RSSI', 'State', 
				'ServingCell_Distance', 'CQI']
            class_names = ['high', 'low', 'medium']
            array = dataset.values
            arrLen = len(dataset.columns) - 1
            X = array[:,0:arrLen]
            y = array[:,arrLen]
            results[base] = {}
            X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.20, random_state=50)
            score = []
            score2 = []
            names = []

            for name, model in adParameters:

                cv_results = cross_val_score(model, X_train, Y_train, cv=skf, scoring='accuracy')
                results[base][name] = {}
                results[base][name]['mean'] = cv_results.mean()
                results[base][name]['std'] = cv_results.std()
                print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))
                mean = results[base][name]['mean']
                score.append(mean)
                names.append(name)
                score2.append(cv_results)
            
            sortArr = sorted(score, reverse=True)
            #fig = pyplot.figure()
            #ax = fig.add_subplot(111)
            #bp = ax.boxplot(score2, labels=names)
            #ax.set_title(f'10-fold Cross-Validation AD {base}') 
            #fig.savefig(f'figs/ad/{base}-ad.png')
            bestValue = sortArr[0]
            acuracias = []
            for name, model in adParameters:
                mean = results[base][name]['mean']
                #if mean == bestValue:
                params = model.get_params()
                pruning = params['ccp_alpha']
                #if pruning > 0:
                #    print(f'{name} {base} com score médio de {mean}. Pruning: {pruning} ')
                #else:
                 #   print(f'{name} {base} com score médio de {bestValue}.')  

                clf = model.fit(X_train, Y_train)
                predictions = model.predict(X_test)
                #fig2 = pyplot.figure(figsize=(25,20)) 
                #_ = tree.plot_tree(clf, feature_names=feature_names,  class_names=class_names, filled=True)
                #fig2.savefig(f'figs/ad/decision-tree-{base}-{name}')
                # Evaluate predictions
                print('Acurácia de predição:', accuracy_score(Y_test, predictions))
                a = accuracy_score(Y_test, predictions)
                acuracias.append(a)
                #print('Matriz de confusão:')
                #print(confusion_matrix(Y_test, predictions))
                #print('Report de classificação:')
                #print(classification_report(Y_test, predictions))#, zero_division=1))
                # Plot non-normalized confusion matrix
                #titles_options = [("Matriz de confusão sem normalização", None), ("Matriz de confusão normalizada", 'true')]
                #class_names = ['high', 'low', 'medium']
                #for title, normalize in titles_options:
                #    disp = plot_confusion_matrix(clf, X_test, Y_test, 
                 #                                   display_labels=class_names, 
                  #                                  cmap=pyplot.cm.Blues, 
                   #                                 normalize=normalize)
                    #disp.ax_.set_title(f'{title} - {base}')
                    #disp.figure_.savefig(f'figs/ad/cm-{base}-{normalize}-{name}.png')
                    #fig.savefig(f'figs/knn/confusion-matrix-{normalize}.png')
            mean = np.mean(acuracias)
            std = np.std(acuracias)
            print(f'Acurácia Média: {mean} ({std})')
            table2_mean = []
            for i in range(1,4):
                for name, model in adParameters:
                    size = i * 10
                    randomState = i * 15
                    X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=size, random_state=randomState)
                    cv_results = cross_val_score(model, X_train, Y_train, cv=skf, scoring='accuracy')
                    print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))
                    mean = cv_results.mean()
                    table2_mean.append(mean)

            mean = np.mean(table2_mean)
            std = np.std(table2_mean)
            
            print(f'3x10-Fold Cross Validation: {mean} ({std})')
        
        #
        #print(f'MediaGeral: {mediaGeral_result}')




    def kNN(self):
        print('----------------------------------------------------------')
        print('Início Experimento kNN - 10-fold cross validation. W -> Com peso. Atributos numéricos escalonados. Treinando modelos.')
        skf = StratifiedKFold(n_splits=10)
        
        knnParameters = []
        kVar = [1, 3, 5, 7, 9, 11, 13]
        for i in kVar:
            knnParameters.append((f'{i}', KNeighborsClassifier(n_neighbors=i)))
            knnParameters.append((f'{i}W', KNeighborsClassifier(n_neighbors=i, weights='distance')))
            
        results = {}

        for base in self.bases:
            print('----------------')
            print(base, 'cross_val_score treinamento')
            print('---------------')
            dataset = self.bases[base]
            array = dataset.values
            arrLen = len(dataset.columns) - 1
            X = array[:,0:arrLen]
            y = array[:,arrLen]
            results[base] = {}
            X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.20, random_state=50)
            score = []
            score2 = []
            names = []
            for name, model in knnParameters:

                cv_results = cross_val_score(model, X_train, Y_train, cv=skf, scoring='accuracy')
                results[base][name] = {}
                results[base][name]['mean'] = cv_results.mean()
                results[base][name]['std'] = cv_results.std()
                print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))
                mean = results[base][name]['mean']
                score.append(mean)
                names.append(name)
                score2.append(cv_results)
            

            sortArr = sorted(score, reverse=True)
            fig = pyplot.figure()
            ax = fig.add_subplot(111)
            bp = ax.boxplot(score2, labels=names)
            #ax.set_title(f'kNN {base}')
            
            #pyplot.boxplot(score2, labels=names)         
            #pyplot.title(f'10-fold Cross-Validation kNN {base}')       
            fig.savefig(f'figs/knn/{base}-kNN.png')
            bestValue = sortArr[0]
            #print(f'{base}: {bestValue} -- {sortArr}')
            acuracias = []
            for name, model in knnParameters:
                mean = results[base][name]['mean']
                if mean == bestValue:
                    print(f'{base} com score médio de {bestValue}. Melhor parâmetro: {name}')
                    t = model.fit(X_train, Y_train)
                    predictions = model.predict(X_test)
                    
                    # Evaluate predictions
                    print('Acurácia de predição:', accuracy_score(Y_test, predictions))
                    #print('Matriz de confusão:')
                    #print(confusion_matrix(Y_test, predictions))
                    #print('Report de classificação:')
                    #print(classification_report(Y_test, predictions))#, zero_division=1))
                    a = accuracy_score(Y_test, predictions)
                    acuracias.append(a)
                    # Plot non-normalized confusion matrix
                    titles_options = [("Matriz de confusão sem normalização", None), ("Matriz de confusão normalizada", 'true')]
                    class_names = ['high', 'low', 'medium']
                    for title, normalize in titles_options:
                        disp = plot_confusion_matrix(t, X_test, Y_test, 
                                                        display_labels=class_names, 
                                                        cmap=pyplot.cm.Blues, 
                                                        normalize=normalize)
                        disp.ax_.set_title(f'{title} - {base}')
                        disp.figure_.savefig(f'figs/knn/cm-{base}-{normalize}.png')
                        #fig.savefig(f'figs/knn/confusion-matrix-{normalize}.png') 
                        #    
            mean = np.mean(acuracias)
            std = np.std(acuracias)
            print(f'Acurácia Média: {mean} ({std})')
            table2_mean = []
            for i in range(1,4):
                for name, model in knnParameters:
                    size = i * 10
                    randomState = i * 15
                    X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=size, random_state=randomState)
                    cv_results = cross_val_score(model, X_train, Y_train, cv=skf, scoring='accuracy')
                    #print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))
                    mean = cv_results.mean()
                    table2_mean.append(mean)

            mean = np.mean(table2_mean)
            std = np.std(table2_mean)        
            print(f'3x10-Fold Cross Validation: {mean} ({std})')

        print('Fim Experimento kNN -')
        print('----------------------------------------------------------')

chkp = Supervisioned('teste.csv')
chkp.generateBases()