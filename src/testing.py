from pandas import read_csv
from pprint import pprint
import numpy as np
import pandas as pd
import scikit_posthocs as sp
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
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
from pre_processing.base import BaseReduction
from sklearn.metrics import plot_confusion_matrix
from sklearn import tree
from sklearn.preprocessing import normalize
from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.metrics import pairwise_distances
from sklearn.metrics import davies_bouldin_score
from sklearn.cluster import AgglomerativeClustering
from sklearn import preprocessing
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LassoCV
from sklearn.linear_model import RidgeCV
from random import randint
from scipy import stats
from scipy.stats import wilcoxon

class StatsTesting:

    def __init__(self, scenario):
        self.path = path.abspath(getcwd())
        self.datasetPath = path.join(self.path, 'data', 'modified', scenario)
        self.BaseReduzida1 = None
        self.BaseReduzida2 = None
        self.BaseReduzida3 = None
        self.BaseOriginal = None
        self.bases = None
        self.baggingTable = pd.DataFrame({"Estratégia": [], "10": [], "15": [], "20": [], "Média (class)": []})
        self.boostingTable = pd.DataFrame({"Estratégia": [], "10": [], "15": [], "20": [], "Média (class)": []})
        self.stackingHomTable = pd.DataFrame({"Estratégia": [], "10": [], "15": [], "20": [], "Média (class)": []})
        self.stackingHetTable = pd.DataFrame({"Configuração": [], "10": [], "15": [], "20": [], "Média (class)": []})
        self.featureSelComite = pd.DataFrame({"Estratégia": [], "10": [], "15": [], "20": [], "Média (class)": []})
        self.baggingpValueT = pd.DataFrame({"Estratégia": [], "10": [], "15": [], "20": []})
        self.baggingpValueC = pd.DataFrame({"Estratégia": [], "10": [], "15": [], "20": []})
        self.measurements1 = {}
        self.measurements2 = {}
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
   

    def runSupervisionedModels(self):
        measurements = {}
        skf = StratifiedKFold(n_splits=10)
        models = [
            ('k-NN', KNeighborsClassifier(n_neighbors=13, weights='distance')),
            ('AD', DecisionTreeClassifier()),
            ('NB', GaussianNB()),
            ('MLP', MLPClassifier(momentum=0.8, max_iter=500, learning_rate_init=0.1, hidden_layer_sizes=12))
        ]
    
        
        
        for base in self.bases:
            measurements[base] = {}
            print('='*20 + base + '='*20)
            dataset = self.bases[base]
            array = dataset.values
            arrLen = len(dataset.columns) - 1
            X = array[:,0:arrLen]
            y = array[:,arrLen]
            X = normalize(X)
            le = preprocessing.LabelEncoder()
            y = le.fit_transform(y)
            X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.20, random_state=50)
            for name, model in models:
                
                clf = model.fit(X_train, Y_train)
                predictions = model.predict(X_test)    
                print(f'{name} Acurácia de predição:', accuracy_score(Y_test, predictions))
                measurements[base][name] = predictions
                #measurements[base][name].append(predictions)
                #a = accuracy_score(Y_test, predictions)
            
        
        pprint(measurements)
        for base in self.bases:
            msAD = measurements[base]['AD']
            msknn = measurements[base]['k-NN']
            msNB = measurements[base]['NB']
            msMLP = measurements[base]['MLP']
            #print(msAD)
            mean1 = np.mean(msAD)
            mean2 = np.mean(msknn)
            mean3 = np.mean(msNB)
            mean4 = np.mean(msMLP)
            
            data = [
                ('AD', msAD),
                ('knn', msknn),
                ('NB', msNB),
                ('MLP', msMLP)
            ]
            possibilities = [
                (msAD, msknn),
                (msAD, msNB),
                (msAD, msMLP),
                (msknn, msNB),
                (msknn, msMLP),
                (msNB, msMLP)
            ]
 
            print('='*10 + base + '='*10 + '\n')
            for a, b in possibilities:
            
                tests = [
                    ('Ttest-ind', stats.ttest_ind(a, b)),
                    ('Wilcoxon', stats.wilcoxon(a, b))
                ]
                for name, test in tests:
                    stat, p = test
                    if p > 0.05:
	                    print(f'{name}: stat=%.3f, p=%.3f -- Probably the same distribution' % (stat, p))
                    else:
                        print(f'{name}: stat=%.3f, p=%.3f -- Probably different distribution' % (stat, p))
	                    #print('Probably different distributions')
            
            print('\n')
    
    def runModels(self):
        measurements = {}
        skf = StratifiedKFold(n_splits=10)
        
        models = [
            ('aglomerative', AgglomerativeClustering(n_clusters=13)),
            ('kmeans', KMeans(n_clusters=13))
        ]
    
        
        
        for base in self.bases:
            measurements[base] = {}
            print('='*20 + base + '='*20)
            dataset = self.bases[base]
            array = dataset.values
            arrLen = len(dataset.columns) - 1
            X = array[:,0:arrLen]
            y = array[:,arrLen]
            X = normalize(X)
            le = preprocessing.LabelEncoder()
            y = le.fit_transform(y)
            X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.40, random_state=42)
            for name, model in models:
                
                clf = model.fit(X_train)
                cr_index = metrics.adjusted_rand_score(Y_train, clf.labels_)
                measurements[base][name] = clf.labels_
        
        print(measurements)
        for base in self.bases:
            msAG = measurements[base]['aglomerative']
            msKmeans = measurements[base]['kmeans']
           
            
            data = [
                ('aglomerative', msAG),
                ('kmeans', msKmeans)
            ]
            possibilities = [
                (msAG, msKmeans)
            ]
 
            print('='*10 + base + '='*10 + '\n')
            for a, b in possibilities:
            
                tests = [
                    ('Ttest-ind', stats.ttest_ind(a, b)),
                    ('Wilcoxon', stats.wilcoxon(a, b))
                ]
                for name, test in tests:
                    stat, p = test
                    if p > 0.05:
	                    print(f'{name}: stat=%.3f, p=%.3f -- Probably the same distribution' % (stat, p))
                    else:
                        print(f'{name}: stat=%.3f, p=%.3f -- Probably different distribution' % (stat, p))
	                    #print('Probably different distributions')
            
            print('\n')

            

chkp = StatsTesting('teste.csv')
chkp.generateBases()
#chkp.runSupervisionedModels()
chkp.runModels()
#chkp.kNN()
#chkp.step1()
#chkp.step2()
#chkp.step3()
#chkp.step4()
#chkp.final_test()