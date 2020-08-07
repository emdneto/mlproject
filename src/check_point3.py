from pandas import read_csv
from pandas.plotting import scatter_matrix
from matplotlib import pyplot
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
from sklearn.naive_bayes import GaussianNB
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
from pprint import pprint
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
from pprint import pprint
from pre_processing.base import BaseReduction
from sklearn.metrics import plot_confusion_matrix

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

        self.kNN()

    def kNN(self):
        print('----------------------------------------------------------')
        print('Início Experimento kNN - 10-fold cross validation. W -> Com peso. Atributos numéricos escalonados. Treinando modelos.')
        skf = StratifiedKFold(n_splits=10)
        
        knnParameters = []
        kVar = [1, 3, 5]
        for i in kVar:
            knnParameters.append((f'kNN-{i}', KNeighborsClassifier(n_neighbors=i)))
            knnParameters.append((f'kNN-{i}-W', KNeighborsClassifier(n_neighbors=i, weights='distance')))
            
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
            ax.set_title(f'10-fold Cross-Validation kNN {base}')
            #pyplot.boxplot(score2, labels=names)         
            #pyplot.title(f'10-fold Cross-Validation kNN {base}')       
            fig.savefig(f'figs/knn/{base}-kNN.png')
            bestValue = sortArr[0]
            #print(f'{base}: {bestValue} -- {sortArr}')
            print(score2)
            for name, model in knnParameters:
                mean = results[base][name]['mean']
                if mean == bestValue:
                    print(f'{base} com score médio de {bestValue}. Melhor parâmetro: {name}')
                    t = model.fit(X_train, Y_train)
                    predictions = model.predict(X_test)
                    
                    # Evaluate predictions
                    print('Acurácia de predição:', accuracy_score(Y_test, predictions))
                    print('Matriz de confusão:')
                    print(confusion_matrix(Y_test, predictions))
                    print('Report de classificação:')
                    print(classification_report(Y_test, predictions))#, zero_division=1))
                    # Plot non-normalized confusion matrix
                    titles_options = [("Matriz de confusão sem normalização", None), ("Matriz de confusão normalizada", 'true')]
                    class_names = ['high', 'low', 'medium']
                    for title, normalize in titles_options:
                        disp = plot_confusion_matrix(t, X_test, Y_test, 
                                                        display_labels=class_names, 
                                                        cmap=pyplot.cm.Blues, 
                                                        normalize=normalize)
                        disp.ax_.set_title(title)
                        disp.figure_.savefig(f'figs/knn/cm-{base}-{normalize}.png')
                        #fig.savefig(f'figs/knn/confusion-matrix-{normalize}.png')


            
            
        print('Fim Experimento kNN -')
        print('----------------------------------------------------------')

            #sort_orders = sorted(results[base].items(), key=lambda x: x[1], reverse=True)

                





chkp = Supervisioned('teste.csv')
chkp.generateBases()