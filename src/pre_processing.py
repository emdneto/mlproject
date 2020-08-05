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


class PreProcessing:

    def __init__(self, scenario):
        self.path = path.abspath(getcwd())
        self.datasetPath = path.join(self.path, 'data', 'modified', scenario)
    
    def run(self):
     #   names = ['Longitude', 'Latitude', 'Speed', 'Operatorname', 'CellID', 
	#			'RSRP', 'RSRQ', 'SNR', 'RSSI', 'State', 'ServingCell_Lon', 'ServingCell_Lat', 
	#			'ServingCell_Distance', 'CQI']

        names = ['Speed', 'Operatorname', 'CellID', 
				'RSRP', 'RSRQ', 'SNR', 'RSSI', 'State', 
				'ServingCell_Distance', 'CQI']

        
        dataset = read_csv(self.datasetPath, usecols=names)
        #print(dataset.describe())
        #with open('descibre.txt', 'w') as f:
         #   f.write(str(dataset.describe()))
         #   f.close()
        
        #print(dataset.groupby('CQI').size())
        
        label_encoder_X = LabelEncoder()
        dataset['Operatorname'] = label_encoder_X.fit_transform(dataset['Operatorname'])
        dataset['State'] = label_encoder_X.fit_transform(dataset['State'])

        array = dataset.values
        f1 = len(names) - 2
        f2 = len(names) - 1
        #Separando variáveis independentes e dependentes
        X = array[:,0:9]
        y = array[:,f2]

        
        imputer = SimpleImputer(missing_values = np.nan, strategy = "most_frequent", verbose = 0)
        imputer = imputer.fit(X)
        X = imputer.transform(X)
        X_train, X_validation, Y_train, Y_validation = train_test_split(X, y, test_size=0.20, random_state=50)

        print(X_train.shape)
    
        print(X_validation.shape)
        
        df = pd.DataFrame(data=X_train, columns=['Speed', 'Operatorname', 'CellID', 'RSRP', 'RSRQ', 'SNR', 'RSSI', 'State', 'ServingCell_Distance'])
        
        print(df.describe())
        #stats.describe(X_validation)

        #ss_X = StandardScaler()
        #X = ss_X.fit_transform(X)
        #X_validation = ss_X.transform(X_validation)
     #PCA
        print('---------------- PCA -------------------')
        print(X.shape)
        pca = PCA(svd_solver='arpack')
        pca.fit(X)
        teste = pca.transform(X)
        print(teste.shape)
        print(pca.explained_variance_ratio_)
         #PCA
        print('---------------- SVD -------------------')
        print(X.shape)
        svd = TruncatedSVD()
        svd.fit(X)
        teste = svd.transform(X)
        print(teste.shape)
        print(svd.explained_variance_ratio_)

        print('---------------- LDA -------------------')
        print(X.shape)
        lda = LinearDiscriminantAnalysis()
        lda.fit(X,y)
        teste = lda.transform(X)
        print(teste.shape)
        print(lda.explained_variance_ratio_)




teste = PreProcessing('teste.csv')
teste.run()