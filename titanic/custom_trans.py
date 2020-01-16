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