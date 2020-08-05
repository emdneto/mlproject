from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from pandas import read_csv
import numpy as np
import pandas as pd

class BaseReduction:

    def __init__(self):
        self.dataset = 0
    

    def original(self, dataset):
        array = dataset.values
        X = array[:,0:9]
        y = array[:,9]
        imputer = SimpleImputer(missing_values = np.nan, strategy = "most_frequent", verbose = 0)
        imputer = imputer.fit(X)
        X = imputer.transform(X)
        ss_X = StandardScaler()
        X = ss_X.fit_transform(X)
        df = pd.DataFrame(data=X)
        df['class'] = y
        return df

    def reduction1(self, dataset):
        dataset = dataset.loc[dataset['Speed'] <= 5]
        array = dataset.values
        X = array[:,0:9]
        y = array[:,9]
        imputer = SimpleImputer(missing_values = np.nan, strategy = "most_frequent", verbose = 0)
        imputer = imputer.fit(X)
        X = imputer.transform(X)
        ss_X = StandardScaler()
        X = ss_X.fit_transform(X)

        df = pd.DataFrame(data=X)
        df['class'] = y
        return df

    def reduction2(self, dataset):
        array = dataset.values
        X = array[:,0:9]
        y = array[:,9]
        imputer = SimpleImputer(missing_values = np.nan, strategy = "most_frequent", verbose = 0)
        imputer = imputer.fit(X)
        X = imputer.transform(X)
        ss_X = StandardScaler()
        X = ss_X.fit_transform(X)
        #print(X.shape)
        clf = ExtraTreesClassifier(n_estimators=50)
        clf = clf.fit(X, y)
        #print(clf.feature_importances_)
        model = SelectFromModel(clf, prefit=True)
        X_new = model.transform(X)
        #print(X_new.shape)
        df = pd.DataFrame(data=X_new)
        df['class'] = y

        return df
    
    def reduction3(self, dataset):
        array = dataset.values
        X = array[:,0:9]
        y = array[:,9]
        imputer = SimpleImputer(missing_values = np.nan, strategy = "most_frequent", verbose = 0)
        imputer = imputer.fit(X)
        X = imputer.transform(X)
        ss_X = StandardScaler()
        X = ss_X.fit_transform(X)
        #print(X.shape)
        lda = LinearDiscriminantAnalysis()
        lda.fit(X,y)
        #print(clf.feature_importances_)
        teste = lda.transform(X)
        #print(teste.shape)
        df = pd.DataFrame(data=teste)
        df['class'] = y

        #print(df)
        return df
    
        #print(df.shape)

        #df = pd.DataFrame(data=[X_new, y])
        #print(df.shape)
        #print(np.concatenate((X_new, y))
        #return [[X_new, y]]
        #return [X_new, ]


        




