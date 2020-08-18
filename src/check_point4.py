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
from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.metrics import pairwise_distances
from sklearn.metrics import davies_bouldin_score
from sklearn.cluster import AgglomerativeClustering
class Clustering:

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
        
        
    def kmeans(self):
        print("="*50)
        print('Iniciando experimento k-means')
        print("="*50, '\n')
        base = 'BaseReduzida1'
        print('='*10, base, '='*10)
        dataset = self.bases[base]
        array = dataset.values
        arrLen = len(dataset.columns) - 1
        X = array[:,0:arrLen]
        y = array[:,arrLen]
        X = normalize(X)
        
        ks = list(range(2, 21))
        
        metric_names = ['DB-Index', 'Silhouette-Score', 'AR-Index']
        
        scores = {
            'DB-Index': [],
            'Silhouette-Score': [],
            'AR-Index': []
        }
        
        
        X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.30, random_state=50)
        for k in ks:
            k_means_model = KMeans(n_clusters=k, n_init=5).fit(X_train)
            labels = k_means_model.labels_
            clusters = k_means_model.cluster_centers_
            print(clusters)
            predict_labels = k_means_model.predict(X_test)
            
            
            db_index = davies_bouldin_score(X_test, predict_labels)
            scores['DB-Index'].append(db_index)
            silhouette_score = metrics.silhouette_score(X_test, predict_labels, metric='euclidean')
            scores['Silhouette-Score'].append(silhouette_score)
            labels_copy = labels[4136:7240]
            cr_index = metrics.adjusted_rand_score(labels_copy, predict_labels)
            scores['AR-Index'].append(cr_index)
            print('='*10)
            print('Valor de K:', k)
            print('Indíce DB:', db_index)
            print('Silhouette Score:', silhouette_score)
            print('Adjusted-Rand:', cr_index)
            #print('labels:', labels)
            #print('predicted labels', predict_labels)
            #print('CR-Index:', cr_index)
            
            
        
        print('='*20, "Calculando Média e desvio padrão para cada índice", '='*20)
        for metric in metric_names:
            print(metric)
            score = scores[metric]
            mean = np.mean(score)
            std = np.std(score)
            print(f'Média: {mean} ({std})')
            #print(score)
            fig, ax = pyplot.subplots()
            ax.plot(ks, score)
            ax.set_xticks(ks)
            ax.set(xlabel='Valor de K', ylabel=metric)
            ax.grid()
            fig.savefig(f"figs/checkpoint4/kmeans/{metric}.png")
            pyplot.show()
        #pyplot.show()

        
    
        
    
    def agglomerativeClustering(self):
        print("="*50)
        print('Iniciando experimento Hierárquico Aglomerativo')
        print("="*50, '\n')
        base = 'BaseReduzida1'
        print('='*10, base, '='*10)
        dataset = self.bases[base]
        array = dataset.values
        arrLen = len(dataset.columns) - 1
        X = array[:,0:arrLen]
        y = array[:,arrLen]
        X = normalize(X)
        
        ks = list(range(2, 21))
        
        metric_names = ['DB-Index', 'Silhouette-Score', 'AR-Index']
        
        scores = {
            'DB-Index': [],
            'Silhouette-Score': [],
            'AR-Index': []
        }
        
        
        X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.30, random_state=50)
        for k in ks:
            model = AgglomerativeClustering(n_clusters=k).fit(X_train)
            labels = model.labels_
            clusters = model.n_clusters_
            
            predict_labels = model.fit_predict(X_test)
            
            
            db_index = davies_bouldin_score(X_test, predict_labels)
            scores['DB-Index'].append(db_index)
            silhouette_score = metrics.silhouette_score(X_test, predict_labels, metric='euclidean')
            scores['Silhouette-Score'].append(silhouette_score)
            labels_copy = labels[4136:7240]
            cr_index = metrics.adjusted_rand_score(labels_copy, predict_labels)
            scores['AR-Index'].append(cr_index)
            print('='*10)
            print('Valor de K:', k)
            print('Indíce DB:', db_index)
            print('Silhouette Score:', silhouette_score)
            print('Adjusted-Rand:', cr_index)
            #print('labels:', labels)
            #print('predicted labels', predict_labels)
            #print('CR-Index:', cr_index)
            
            
        
        print('='*20, "Calculando Média e desvio padrão para cada índice", '='*20)
        for metric in metric_names:
            print(metric)
            score = scores[metric]
            mean = np.mean(score)
            std = np.std(score)
            print(f'Média: {mean} ({std})')
            #print(score)
            fig, ax = pyplot.subplots()
            ax.plot(ks, score)
            ax.set_xticks(ks)
            ax.set(xlabel='Valor de K', ylabel=metric)
            ax.grid()
            fig.savefig(f"figs/checkpoint4/agglomerative/{metric}.png")
            pyplot.show()
        #pyplot.show()
    
    
chkp = Clustering('teste.csv')
chkp.generateBases()
#chkp.kmeans()
chkp.agglomerativeClustering()