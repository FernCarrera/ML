from sklearn.base import BaseEstimator,TransformerMixin
import numpy as np
import pandas as pd
import pdb
class AttributesManager(BaseEstimator,TransformerMixin):

    #def __init__(self):
    #    pass
    # required to be duck-taped to transform pipeline
    def fit(self,X,y=None):
        return self
   #s pdb.set_trace()
    def transform(self,X,y=None):
        #print(X)
        #X.info()
        new = X.drop(["Name","Ticket","Survived","Cabin"],axis=1)
        new["Embarked"].fillna('S',inplace=True)

        new.info()
    
        #nan_loc = np.argwhere(np.isnan(X))
        #print(nan_loc)
        return new


class DataFrameSelector(BaseEstimator,TransformerMixin):
    def __init__(self,attribute_names):
        self.attribute_names = attribute_names

    def fit(self,X,y=None):
        return self

    def transform(self,X):
        return X[self.attribute_names]


class MostFrequentImputer(BaseEstimator,TransformerMixin):
    def fit(self,X,y=None):
        self.most_frequent_ = pd.Series([X[c].value_counts().index[0] for c in X],
                                        index=X.columns)
        return self 
    def transform(self,X,y=None):
        return X.fillna(self.most_frequent_)