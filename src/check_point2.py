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

class PreProcessing:

    def __init__(self, scenario):
        self.path = path.abspath(getcwd())
        self.datasetPath = path.join(self.path, 'data', 'modified', scenario)

    def featureExtraction(self):
        names = ['Speed', 'Operatorname', 'CellID', 
				'RSRP', 'RSRQ', 'SNR', 'RSSI', 'State', 
				'ServingCell_Distance', 'CQI']

        dataset = read_csv(self.datasetPath, usecols=names)
        #dataset.hist()
        pyplot.show()
        print(dataset.groupby('CQI').size())
        
        label_encoder_X = LabelEncoder()
        dataset['Operatorname'] = label_encoder_X.fit_transform(dataset['Operatorname'])
        dataset['State'] = label_encoder_X.fit_transform(dataset['State'])
        
        array = dataset.values
        #print(array)
        #f1 = len(names) - 2
        #f2 = len(names) - 1
        #Separando variáveis independentes e dependentes
        X = array[:,0:9]
        y = array[:,9]

        
        imputer = SimpleImputer(missing_values = np.nan, strategy = "most_frequent", verbose = 0)
        imputer = imputer.fit(X)
        X = imputer.transform(X)

        #X_train, X_validation, Y_train, Y_validation = train_test_split(X, y, test_size=0.20, random_state=50)

        #print(X.shape)
        ss_X = StandardScaler()
        X = ss_X.fit_transform(X)
        print(X.shape)
        clf = ExtraTreesClassifier(n_estimators=50)
        clf = clf.fit(X, y)
        print(clf.feature_importances_)
        model = SelectFromModel(clf, prefit=True)
        
        X_new = model.transform(X)
        # X_new = SelectKBest(chi2, k=2).fit_transform(X, y)
        print(X_new.shape)

    def featureSelection(self):
        names = ['Speed', 'Operatorname', 'CellID', 
				'RSRP', 'RSRQ', 'SNR', 'RSSI', 'State', 
				'ServingCell_Distance', 'CQI']

        dataset = read_csv(self.datasetPath, usecols=names)
        #dataset.hist()
        pyplot.show()
        print(dataset.groupby('CQI').size())
        
        label_encoder_X = LabelEncoder()
        dataset['Operatorname'] = label_encoder_X.fit_transform(dataset['Operatorname'])
        dataset['State'] = label_encoder_X.fit_transform(dataset['State'])
        
        array = dataset.values
        #print(array)
        #f1 = len(names) - 2
        #f2 = len(names) - 1
        #Separando variáveis independentes e dependentes
        X = array[:,0:9]
        y = array[:,9]

        
        imputer = SimpleImputer(missing_values = np.nan, strategy = "most_frequent", verbose = 0)
        imputer = imputer.fit(X)
        X = imputer.transform(X)

        #X_train, X_validation, Y_train, Y_validation = train_test_split(X, y, test_size=0.20, random_state=50)

        #print(X.shape)
        ss_X = StandardScaler()
        X = ss_X.fit_transform(X)
        print(X.shape)
        
        clf = ExtraTreesClassifier(n_estimators=50)
        clf = clf.fit(X, y)
        print(clf.feature_importances_)
        model = SelectFromModel(clf, prefit=True)

        X_new = model.transform(X)
        # X_new = SelectKBest(chi2, k=2).fit_transform(X, y)
        print(X_new.shape)

        print('---------------- LDA -------------------')
        print(X.shape)
        lda = LinearDiscriminantAnalysis()
        lda.fit(X,y)
        teste = lda.transform(X)
        print(teste.shape)
        print(lda.explained_variance_ratio_)
    
    def reduction(self):
     #   names = ['Longitude', 'Latitude', 'Speed', 'Operatorname', 'CellID', 
	#			'RSRP', 'RSRQ', 'SNR', 'RSSI', 'State', 'ServingCell_Lon', 'ServingCell_Lat', 
	#			'ServingCell_Distance', 'CQI']

        names = ['Speed', 'Operatorname', 'CellID', 
				'RSRP', 'RSRQ', 'SNR', 'RSSI', 'State', 
				'ServingCell_Distance', 'CQI']

        
        dataset = read_csv(self.datasetPath, usecols=names)
        
        print(dataset.groupby('CQI').size())

        #print(dataset.describe())
        #with open('descibre.txt', 'w') as f:
         #   f.write(str(dataset.describe()))
         #   f.close()
        
        #print(dataset.groupby('CQI').size())
        print(dataset.shape)
        dataset = dataset[dataset['Speed'] <= 5] 
        print(dataset.shape)

        label_encoder_X = LabelEncoder()
        dataset['Operatorname'] = label_encoder_X.fit_transform(dataset['Operatorname'])
        dataset['State'] = label_encoder_X.fit_transform(dataset['State'])
        
        testando = True
        #df_filtered = df[df['Age'] >= 25] 
        #if testando:
        
        
        #dataset = dataset[dataset['CQI'] != 'NaN']

       # print(dataset.shape)

        array = dataset.values
        #print(array)
        f1 = len(names) - 2
        f2 = len(names) - 1
        #Separando variáveis independentes e dependentes
        X = array[:,0:9]
        y = array[:,9]

        
        imputer = SimpleImputer(missing_values = np.nan, strategy = "most_frequent", verbose = 0)
        imputer = imputer.fit(X)
        X = imputer.transform(X)

        X_train, X_validation, Y_train, Y_validation = train_test_split(X, y, test_size=0.20, random_state=50)

        #print(X.shape)
        ss_X = StandardScaler()
        X = ss_X.fit_transform(X)
		
        models = []
        models.append(('KNN', KNeighborsClassifier(3)))
        models.append(('CART', DecisionTreeClassifier()))
        
        results = []
        names = []
      
        for name, model in models:
            kfold = StratifiedKFold(n_splits=2)
            cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy')
            results.append(cv_results)
            names.append(name)
            print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))

		# Compare Algorithms
        pyplot.boxplot(results, labels=names)
        pyplot.title('Algorithm Comparison')
        pyplot.show()

		# Make predictions on validation dataset
        #model = RandomForestClassifier()
        #model.fit(X, Y_train)
        #predictions = model.predict(X_validation)
        # Evaluate predictions
        #print(accuracy_score(Y_validation, predictions))
        #print(confusion_matrix(Y_validation, predictions))
        #print(classification_report(Y_validation, predictions, zero_division=1))

teste = PreProcessing('teste.csv')
#teste.reduction()
teste.featureSelection()