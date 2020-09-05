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


class BaggingBoosting:

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
        self.measurementsStack = {}
        self.baggingComparison1 = {}
        self.baggingComparison2 = {}
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

      
    
    def step1(self):
        base = 'BaseReduzida1'
        dataset = self.bases[base]
        array = dataset.values
        arrLen = len(dataset.columns) - 1
        X = array[:,0:arrLen]
        y = array[:,arrLen]
        X = normalize(X)
        le = preprocessing.LabelEncoder()
        y = le.fit_transform(y)
        X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.30, random_state=42)
        
        
        n_estimators = [10, 15, 20]
        
        
        base_estimators = [
            
            ('k-NN', KNeighborsClassifier(n_neighbors=13)),
            ('AD', DecisionTreeClassifier()),
            ('NB', GaussianNB()),
            ('MLP', MLPClassifier(momentum=0.8, max_iter=500, learning_rate_init=0.1, hidden_layer_sizes=12))
        ]
        '''
        base_estimators = [
            
            ('k-NN', KNeighborsClassifier(n_neighbors=13)),
            ('AD', DecisionTreeClassifier()),
            ('NB', GaussianNB())
          
        ]
        '''
        
        measurements = {}
        
        index = 0
        for name, clf in base_estimators:
            self.baggingTable.at[index, "Estratégia"] = name
            self.boostingTable.at[index, "Estratégia"] = name
            self.baggingpValueT.at[index, "Estratégia"] = name
            measurements[name] = {}
            baggingScores = []            
            boostingScores = []
            self.baggingComparison1[name] = {}
            for i in n_estimators:
                
                measurements[name][i] = {}
                print(f'Running {clf} with n_estimators {i}')
                rng = np.random.RandomState(42)
                bagging_clf = BaggingClassifier(base_estimator=clf, n_estimators=i, random_state=rng).fit(X_train, Y_train)
                boosting_clf = GradientBoostingClassifier(init=clf, n_estimators=i, random_state=rng).fit(X_train, Y_train)
                

                bagging_predict = bagging_clf.predict(X_test)
                boosting_predict = boosting_clf.predict(X_test)
                self.baggingComparison1[name][i] = bagging_predict
                measurements[name][i]['boosting'] = boosting_predict
                measurements[name][i]['bagging'] = bagging_predict
                '''
                ttest = stats.kruskal(Y_test, bagging_predict)
                print(f'test for {name} with {i} estimators: {ttest}')
                ss = "{:.4f}".format(ttest[1]) 
                print(ss)
                self.baggingpValueT.at[index, str(i)] = ss
                '''
                bagging_acc = accuracy_score(Y_test, bagging_predict)
                boosting_acc = accuracy_score(Y_test, boosting_predict)
                
                '''
                print(f'Acurácia de predição - Bagging ({name}):', bagging_acc)
                print(f'Acurácia de predição - Boosting ({name}):', boosting_acc)
                '''
                formatBagging = bagging_acc * 100
                formatBoosting = boosting_acc * 100
                #self.measurements1[name][i] = bagging_predict
                baggingScores.append(formatBagging)
                boostingScores.append(formatBoosting)
                
                self.baggingTable.at[index, str(i)] =   formatBagging
                self.boostingTable.at[index, str(i)] =   formatBoosting

            self.baggingTable.at[index, "Média (class)"] = np.mean(baggingScores)
            self.boostingTable.at[index, "Média (class)"] = np.mean(boostingScores)
            index += 1
        
        
        ''''
        scores = []
        for name, clf in base_estimators:
            score = measurements[name][15]
            scores.append(score)
        
        test = stats.kruskal(scores[0], scores[1], scores[2], scores[3])
        print(test)
        '''
        self.baggingTable.at[index+1, "Estratégia"] = 'Média (TAM)'
        self.baggingTable.at[index+1, "10"] = self.baggingTable['10'].mean()
        self.baggingTable.at[index+1, "15"] = self.baggingTable['15'].mean()
        self.baggingTable.at[index+1, "20"] = self.baggingTable['20'].mean()
        self.baggingTable.at[index+1, "Média (class)"] = self.baggingTable['Média (class)'].mean()
        
        self.boostingTable.at[index+1, "Estratégia"] = 'Média (TAM)'
        self.boostingTable.at[index+1, "10"] = self.boostingTable['10'].mean()
        self.boostingTable.at[index+1, "15"] = self.boostingTable['15'].mean()
        self.boostingTable.at[index+1, "20"] = self.boostingTable['20'].mean()
        self.boostingTable.at[index+1, "Média (class)"] = self.boostingTable['Média (class)'].mean()
        
        
        print('Bagging Table \n')
        print(self.baggingTable.head(5))              
        
        print('\n')
        
        print('Boosting Table')
        print(self.boostingTable.head(5))
        
        print('p-value: Bagging x Boosting for all estimators and n_estimators (10, 15, 20)')
        for name, clf in base_estimators:
            for i in n_estimators:
                ms1 = measurements[name][i]['bagging']
                ms2 = measurements[name][i]['boosting']
                tests = [
                    ('Ttest-ind', stats.ttest_ind(ms1, ms2)),
                    ('Wilcoxon', stats.wilcoxon(ms1, ms2))
                ]
                for method, test in tests:
                    stat, p = test
                    if p > 0.05:
	                    print(f'{method}-{name}-{i}: stat=%.3f, p=%.3f -- Probably the same distribution' % (stat, p))
                    else:
                        print(f'{method}-{name}-{i}: stat=%.3f, p=%.3f -- Probably different distribution' % (stat, p))
        
        print('\n')
        #print(self.baggingpValueT)
    
    
    def step2(self):
        base = 'BaseReduzida1'
        dataset = self.bases[base]
        array = dataset.values
        arrLen = len(dataset.columns) - 1
        X = array[:,0:arrLen]
        y = array[:,arrLen]
        X = normalize(X)
        
        
        n_estimators = [10, 15, 20]
        
        base_estimators = [
            
            ('k-NN', KNeighborsClassifier(n_neighbors=13)),
            ('AD', DecisionTreeClassifier()),
            ('NB', GaussianNB()),
            ('MLP', MLPClassifier(momentum=0.8, max_iter=500, learning_rate_init=0.1, hidden_layer_sizes=12))
        ]
        
        estimators = {}
        for name, clf in base_estimators:
            estimators[name] = {}
            estimators[name]['10'] = []
            estimators[name]['15'] = []
            estimators[name]['20'] = []
            
            for n in n_estimators:
                for i in range(n+1):
                    estimators[name][str(n)].append((f'{name}{n}{i}', clf))
        index = 0
        for name, clf in base_estimators:
            self.stackingHomTable.at[index, 'Estratégia'] = name
            self.measurementsStack[name] = {}
            scores = []
            for n in n_estimators:
                self.measurementsStack[name][n] = {}
                rng = np.random.RandomState(42)
                stackingHom = StackingClassifier(estimators=estimators[name][str(n)], final_estimator=RandomForestClassifier(max_depth=5, n_estimators=n, max_features=1))
                X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.30, stratify=y, random_state=rng)
                predictions = stackingHom.fit(X_train, Y_train).predict(X_test)
                self.measurementsStack[name][n]['stackhom'] = predictions
                acc = accuracy_score(Y_test, predictions) * 100
                scores.append(acc)
                print(f'{name} with {n} estimators: {acc}')
                self.stackingHomTable.at[index, str(n)] =   acc
                
            self.stackingHomTable.at[index, "Média (class)"] = np.mean(scores)
            index += 1

        self.stackingHomTable.at[index+1, "Estratégia"] = 'Média (TAM)'
        self.stackingHomTable.at[index+1, "10"] = self.stackingHomTable['10'].mean()
        self.stackingHomTable.at[index+1, "15"] = self.stackingHomTable['15'].mean()
        self.stackingHomTable.at[index+1, "20"] = self.stackingHomTable['20'].mean()
        self.stackingHomTable.at[index+1, "Média (class)"] = self.stackingHomTable['Média (class)'].mean()

        print(self.stackingHomTable.head(5))
    
    def step3(self):
        base = 'BaseReduzida1'
        dataset = self.bases[base]
        array = dataset.values
        arrLen = len(dataset.columns) - 1
        X = array[:,0:arrLen]
        y = array[:,arrLen]
        X = normalize(X)
        
        
        
        '''
        confs = {
            "A": [('AD', DecisionTreeClassifier()), ('k-NN', KNeighborsClassifier(n_neighbors=13)), ],
            "B": [('AD', DecisionTreeClassifier()),('MLP', MLPClassifier(momentum=0.8, max_iter=500, learning_rate_init=0.1, hidden_layer_sizes=12))],
            "C": [('k-NN', KNeighborsClassifier(n_neighbors=13)),('MLP', MLPClassifier(momentum=0.8, max_iter=500, learning_rate_init=0.1, hidden_layer_sizes=12))],
            "D": [('AD', DecisionTreeClassifier()),('k-NN', KNeighborsClassifier(n_neighbors=13)), ('MLP', MLPClassifier(momentum=0.8, max_iter=500, learning_rate_init=0.1, hidden_layer_sizes=12))]
        }
        '''
       
        knn_clf = KNeighborsClassifier(n_neighbors=13)
        ad_clf = DecisionTreeClassifier()
        mlp_clf = MLPClassifier(momentum=0.8, max_iter=500, learning_rate_init=0.1, hidden_layer_sizes=12)
        list_confs = ["A", "B", "C", "D"]
        n_estimators = [10, 15, 20]
        
        confs = {
            "A": {},
            "B": {},
            "C": {},
            "D": {}
        }
        
        for conf in list_confs:
            for n in n_estimators:
                confs[conf][n] = []
                if conf == "A":
                    for i in range(1, n+1):
                        k_nn = (f'knn-{conf}{i}', knn_clf)
                        ad   = (f'ad-{conf}{i}', ad_clf)
                        confs[conf][n].append(k_nn)
                        confs[conf][n].append(ad)
                
                if conf == "B":
                    for i in range(1, n+1):
                        ad   = (f'ad-{conf}{i}', ad_clf)
                        mlp  = (f'mlp-{conf}{i}', mlp_clf)
                        confs[conf][n].append(ad)
                        confs[conf][n].append(mlp)
                        
                        
                if conf == "C":
                    for i in range(1, n+1):
                        k_nn = (f'knn-{conf}{i}', knn_clf)
                        mlp  = (f'mlp-{conf}{i}', mlp_clf)
                        confs[conf][n].append(k_nn)
                        confs[conf][n].append(mlp)
                        
                if conf == "D":
                    for i in range(1, n+1):
                        ad   = (f'ad-{conf}{i}', ad_clf)
                        k_nn = (f'knn-{conf}{i}', knn_clf)
                        mlp  = (f'mlp-{conf}{i}', mlp_clf)
                        confs[conf][n].append(ad)
                        confs[conf][n].append(k_nn)
                        confs[conf][n].append(mlp)
            
       
        index = 0
        for conf_name in list_confs:
            self.stackingHetTable.at[index, 'Configuração'] = conf_name
            self.measurementsStack[conf_name] = {}
            scores = []
            for n in n_estimators:
                self.measurementsStack[conf_name][n] = {}
                rng = np.random.RandomState(42)
                estimators = confs[conf_name][n]
                print('='*50)
                print(conf_name, n)
                pprint(estimators)
                print('='*50)
                stackingHet = StackingClassifier(estimators=estimators, final_estimator=RandomForestClassifier(max_depth=5, n_estimators=n, max_features=1))
                X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.30, stratify=y, random_state=rng)
                predictions = stackingHet.fit(X_train, Y_train).predict(X_test)
                self.measurementsStack[conf_name][n]['stackhet'] = predictions
                acc = accuracy_score(Y_test, predictions) * 100
                scores.append(acc)
                print(f'{conf_name} with {n} estimators: {acc}')
                self.stackingHetTable.at[index, str(n)] =   acc
                
            self.stackingHetTable.at[index, "Média (class)"] = np.mean(scores)
            index += 1

        self.stackingHetTable.at[index+1, "Configuração"] = 'Média (conf)'
        self.stackingHetTable.at[index+1, "10"] = self.stackingHetTable['10'].mean()
        self.stackingHetTable.at[index+1, "15"] = self.stackingHetTable['15'].mean()
        self.stackingHetTable.at[index+1, "20"] = self.stackingHetTable['20'].mean()
        self.stackingHetTable.at[index+1, "Média (class)"] = self.stackingHetTable['Média (class)'].mean()
        print(self.stackingHetTable.head(10))

                
        #pprint(confs)
        #pprint(len(confs["D"][10]))
        #pprint(len(confs["D"][15]))
        #pprint(len(confs["D"][20]))
        '''
        base_estimators = [
            
            ('k-NN', KNeighborsClassifier(n_neighbors=13)),
            ('AD', DecisionTreeClassifier()),
            ('NB', GaussianNB()),
            ('MLP', MLPClassifier(momentum=0.8, max_iter=500, learning_rate_init=0.1, hidden_layer_sizes=12))
        ]
        '''
        
        '''
        index = 0
        scores = []
        for conf_name in list_confs:
            self.stackingHetTable.at[index, 'Configuração'] = conf_name
            
            rng = np.random.RandomState(42)
            stackingHet = StackingClassifier(estimators=confs[conf_name], final_estimator=RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1))
            X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.30, stratify=y, random_state=rng)
            predictions = stackingHet.fit(X_train, Y_train).predict(X_test)
            acc = accuracy_score(Y_test, predictions) * 100
            scores.append(acc)
            print(f'{conf_name}: {acc}')       
            self.stackingHetTable.at[index, "Score"] = acc          
            index += 1
        
        self.stackingHetTable.at[index+1, "Configuração"] = 'Média (CONF)'
        self.stackingHetTable.at[index+1, "Score"] = self.stackingHetTable['Score'].mean()
        print(self.stackingHetTable.head(5))
    
    
       '''   
    def step4(self):
        base = 'BaseReduzida1'
        dataset = self.bases[base]
        array = dataset.values
        arrLen = len(dataset.columns) - 1
        X = array[:,0:arrLen]
        y = array[:,arrLen]
        X = normalize(X)
        le = preprocessing.LabelEncoder()
        y = le.fit_transform(y)

        
        
        
        n_estimators = [10, 15, 20]

        
        
        base_estimators = [
            
            ('k-NN', KNeighborsClassifier(n_neighbors=13)),
            ('AD', DecisionTreeClassifier()),
            ('NB', GaussianNB()),
            ('MLP', MLPClassifier(momentum=0.8, max_iter=500, learning_rate_init=0.1, hidden_layer_sizes=12))
        ]
        '''
        base_estimators = [
            
            ('k-NN', KNeighborsClassifier(n_neighbors=13)),
            ('AD', DecisionTreeClassifier()),
            ('NB', GaussianNB())
          
        ]
        '''
        
        
        
        index = 0
        for name, clf in base_estimators:
            self.featureSelComite.at[index, "Estratégia"] = name
            self.baggingpValueC.at[index, "Estratégia"] = name
            self.baggingComparison2[name] = {}
            scores = []            
            for i in n_estimators:
                
                print(f'Running {clf} with n_estimators {i}')
                rng = np.random.RandomState(42)
                X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.30, stratify=y, random_state=rng)
                bagging_clf = BaggingClassifier(base_estimator=clf, n_estimators=i, random_state=rng, max_features=0.5, bootstrap_features=True).fit(X_train, Y_train)
               

                bagging_predict = bagging_clf.predict(X_test)
                self.baggingComparison2[name][i] = bagging_predict
                #self.measurements2[name][i] = bagging_predict
                '''
                ttest = stats.ttest_rel(Y_test, bagging_predict)
                print(f'test for {name} with {i} estimators: {ttest}')
                ss = "{:.4f}".format(ttest[1]) 
                print(ss)
                self.baggingpValueC.at[index, str(i)] = ss
                '''
                bagging_acc = accuracy_score(Y_test, bagging_predict)
                
                print(f'Acurácia de predição - Bagging ({name}):', bagging_acc)
                
                
                formatBagging = bagging_acc * 100
                
                
                scores.append(formatBagging)
                self.featureSelComite.at[index, str(i)] =   formatBagging
                
            self.featureSelComite.at[index, "Média (class)"] = np.mean(scores)
            index += 1
        
        '''
        scores = []
        for name, clf in base_estimators:
            score = measurements[name][15]
            scores.append(score)
        
        test = stats.kruskal(scores[0], scores[1], scores[2], scores[3])
        print(test)
        '''
        self.featureSelComite.at[index+1, "Estratégia"] = 'Média (TAM)'
        self.featureSelComite.at[index+1, "10"] = self.featureSelComite['10'].mean()
        self.featureSelComite.at[index+1, "15"] = self.featureSelComite['15'].mean()
        self.featureSelComite.at[index+1, "20"] = self.featureSelComite['20'].mean()
        self.featureSelComite.at[index+1, "Média (class)"] = self.featureSelComite['Média (class)'].mean()
        
        print(self.featureSelComite.head(10))
        
        #print('p-value table')
        #print(self.baggingpValueC)
        
    
    def stack_test(self):
        
        n_estimators = [10, 15, 20]

        
        base_estimators = [
            
            ('k-NN', 'A'),
            ('AD', 'B'),
            ('NB', 'C'),
            ('MLP', 'D')
        ]
        '''
        base_estimators = [
            
            ('k-NN', KNeighborsClassifier(n_neighbors=13)),
            ('AD', DecisionTreeClassifier()),
            ('NB', GaussianNB())
          
        ]
        '''
        print('p-value for stack hom vs stack het')
        for name1, name2 in base_estimators:
            print(f'{name1} vs {name2}')
            for n in n_estimators:
                ms1 = self.measurementsStack[name1][n]['stackhom']
                ms2 = self.measurementsStack[name2][n]['stackhet']
                
                print(ms1)
                print(ms2)
                
                tests = [
                    ('Ttest-ind', stats.ttest_ind(ms1,ms2)),
                    ('Wilcoxon', stats.wilcoxon(ms1, ms2))
                ]
                for method, test in tests:
                    stat, p = test
                    if p > 0.05:
	                    print(f'{method}-{i}: stat=%.3f, p=%.3f -- Probably the same distribution' % (stat, p))
                    else:
                        print(f'{method}-{i}: stat=%.3f, p=%.3f -- Probably different distribution' % (stat, p))
        
        print('\n')
                

            

chkp = BaggingBoosting('teste.csv')
chkp.generateBases()
#chkp.step1()
chkp.step2()
chkp.step3()
#chkp.step4()
chkp.stack_test()