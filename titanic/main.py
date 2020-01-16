import pandas as pd
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import OneHotEncoder,StandardScaler,OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from custom_trans import AttributesManager


train_set = pd.read_csv("train.csv")
#train_set_N = train_set.drop("Survived",axis=1)
#train_set_N = train_set_N.drop("Cabin",axis=1)

#train_set_N["Embarked"].fillna('S',inplace=True)
#train_set_N.info()

#print(train_set_N.info())
#train_set_N = train_set_N.select_dtypes(exclude=['object'])



# study correlations
#corr_matrix = train_set.corr()
#print(corr_matrix["Survived"].sort_values(ascending=False))
"""
Survived       1.000000
Fare           0.257307
Parch          0.081629
PassengerId   -0.005007
SibSp         -0.035322
Age           -0.077221
Pclass        -0.338481     1:1st,2:2nd,3:3rd class
Name: Survived, dtype: float64
"""





"""Pipeline to deal with the numerical attributes
"""
# imputer fills out missing items with the average value
num_pipeline = Pipeline([('imputer',SimpleImputer(strategy="mean")),
                        ('std_scaleer',StandardScaler())])

# pipeline that computes the whole data including text items
num_attribs = ["Pclass","Age","SibSp","Parch","Fare"]
cat_attributes = ["Sex","Name","Embarked","Ticket","Survived","PassengerId","Cabin"]
rem = ["Sex","Embarked"]
full_pipeline = ColumnTransformer([
                ("num",num_pipeline,num_attribs),
                ("cat_fill",AttributesManager(),cat_attributes),
                ("cat",OneHotEncoder(),rem)])

# prepared data
titanic_prepared = full_pipeline.fit_transform(train_set)

"""
tree_reg = RandomForestRegressor()
tree_reg.fit(train_prepared,train_labels)

predictions = tree_reg.predict(train_prepared)
tree_mse = mean_squared_error(train_labels,predictions)
tree_rmse = np.sqrt(tree_mse)
print(tree_rmse)
"""



