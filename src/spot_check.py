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
from pprint import pprint
import numpy as np



class CQIPredict:
	
	def __init__(self, scenario):
		super().__init__()
		self.path = path.abspath(getcwd())
		self.datasetPath = path.join(self.path, 'data', 'modified', scenario)

	def run(self):
		names = ['Longitude', 'Latitude', 'Speed', 'Operatorname', 'CellID', 
				'RSRP', 'RSRQ', 'SNR', 'RSSI', 'State', 'ServingCell_Lon', 'ServingCell_Lat', 
				'ServingCell_Distance', 'CQI']
		#names = ['Speed', 'Operatorname', 'CellID', 
		#		'RSRP', 'RSRQ', 'SNR', 'RSSI', 'State', 
		#		'ServingCell_Distance', 'CQI']
		dataset = read_csv(self.datasetPath, usecols=names)
		#print(dataset.shape)
		#print(dataset.head(20))
		#print(dataset.describe())
		print(dataset.groupby('CQI').size())

		#dataset.plot(kind='box', subplots=True, layout=(12,12), sharex=False, sharey=False)
		#pyplot.show()
		#dataset.hist()
		#pyplot.show()
		#scatter_matrix(dataset)
		#pyplot.show()
		label_encoder_X = LabelEncoder()
		dataset['Operatorname'] = label_encoder_X.fit_transform(dataset['Operatorname'])
		dataset['State'] = label_encoder_X.fit_transform(dataset['State'])
		print(dataset.head(5))
		
		
		array = dataset.values
		f1 = len(names) - 2
		f2 = len(names) - 1
		
		#Separando variáveis independentes e dependentes
		X = array[:,0:f1]
		y = array[:,f2]
		

	
		imputer = SimpleImputer(missing_values = np.nan, strategy = "most_frequent", verbose = 0)
		imputer = imputer.fit(X)
		X = imputer.transform(X)

		#ct = ColumnTransformer([("encoder", OneHotEncoder(), [3])], remainder="passthrough")
		#X = np.array(ct.fit_transform(X), dtype=np.float)

		#print(X)
		X_train, X_validation, Y_train, Y_validation = train_test_split(X, y, test_size=0.20, random_state=50)
		ss_X = StandardScaler()
		X_train = ss_X.fit_transform(X_train)
		X_validation = ss_X.transform(X_validation)
		#print('before')
		#print(X_train)

		lda = LinearDiscriminantAnalysis()
		x_train2 = lda.fit_transform(X_train, Y_train)
		#print('after')
		#print(x_train2)
		
		models = []

		models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
		models.append(('LDA', LinearDiscriminantAnalysis()))
		models.append(('KNN', KNeighborsClassifier(3)))
		models.append(('CART', DecisionTreeClassifier()))
		models.append(('NB', GaussianNB()))
		#models.append(('GPC', GaussianProcessClassifier(1.0 * RBF(1.0))))
		models.append(('SVM', SVC(gamma='auto')))
		#models.append(('SVM-2', SVC(gamma=2, C=1)))
		#models.append(('SVM-3', SVC(kernel="linear", C=0.025)))
		models.append(('RF', RandomForestClassifier(n_estimators=100, max_features=3)))
		models.append(('GBC', GradientBoostingClassifier()))
		#models.append(('ABC', AdaBoostClassifier()))
		models.append(('MPLC', MLPClassifier(alpha=1, max_iter=1000)))
		#models.append(('QDA', QuadraticDiscriminantAnalysis()))

		results = []
		names = []
		for name, model in models:
			kfold = StratifiedKFold(n_splits=2)
			cv_results = cross_val_score(model, x_train2, Y_train, cv=kfold, scoring='accuracy')
			results.append(cv_results)
			names.append(name)
			print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))

		# Compare Algorithms
		pyplot.boxplot(results, labels=names)
		pyplot.title('Algorithm Comparison')
		pyplot.show()

		# Make predictions on validation dataset
		model = RandomForestClassifier()
		model.fit(X_train, Y_train)
		predictions = model.predict(X_validation)

		# Evaluate predictions
		print(accuracy_score(Y_validation, predictions))
		print(confusion_matrix(Y_validation, predictions))
		print(classification_report(Y_validation, predictions, zero_division=1))



t = CQIPredict('train.csv')
t.run()
#path = "/home/ws/workspace/mestrado/mlproject/data/dataset/static/B_2018.02.12_16.14.01.csv"
#path = "/home/ws/workspace/mestrado/mlproject/data/modified/static.csv"
#names = ['Longitude', 'Latitude', 'Speed', 'CellID', 
#'RSRP', 'RSRQ', 'SNR', 'RSSI', 'ServingCell_Lon', 'ServingCell_Lat', 
#'ServingCell_Distance', 'CQI']

#dataset = read_csv(path, usecols=names)

#print(dataset.head(10))
#array = dataset.values
#X = array[:,0:10]
#y = array[:,11]


#imputer = SimpleImputer(missing_values = np.nan, strategy = "mean", verbose = 0)
#imputer = imputer.fit(X)
#X = imputer.transform(X)

# Spot Check Algorithms
'''
models = []
models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC(gamma='auto')))
models.append(('RF', RandomForestClassifier(n_estimators=100, max_features=3)))
models.append(('GBC', GradientBoostingClassifier()))
models.append(('ABC', AdaBoostClassifier()))
# evaluate each model in turn
results = []
names = []
for name, model in models:
	kfold = StratifiedKFold(n_splits=2, random_state=0, shuffle=True)
	cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy')
	results.append(cv_results)
	names.append(name)
	print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))

# Compare Algorithms
pyplot.boxplot(results, labels=names)
pyplot.title('Algorithm Comparison')
pyplot.show()

# Make predictions on validation dataset
model = RandomForestClassifier()
model.fit(X_train, Y_train)
predictions = model.predict(X_validation)

# Evaluate predictions
print(accuracy_score(Y_validation, predictions))
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions, zero_division=1))
# shape
#print(dataset.shape)
# head
#print(dataset.head(20))
# descriptions
#print(dataset.describe())
# class distribution
#print(dataset.groupby('').size())
'''