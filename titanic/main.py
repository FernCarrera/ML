import pandas as pd
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import OneHotEncoder,StandardScaler,OrdinalEncoder
from sklearn.pipeline import Pipeline,FeatureUnion
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
#from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
from custom_trans import AttributesManager,DataFrameSelector,MostFrequentImputer
from sklearn.svm import SVC
import csv

train_set = pd.read_csv("train.csv")
test_set = pd.read_csv("test.csv")

num_atts = ["Age","SibSp","Parch","Fare"]
# Numerical pipeline
num_pipeline = Pipeline([
                ("Select_numeric",DataFrameSelector(num_atts)),
                ("imputer",SimpleImputer(strategy="median"))
])

num = num_pipeline.fit_transform(train_set)

cat_atts = ["Pclass","Sex","Embarked"]

cat_pipeline = Pipeline([
            ("Select_cat",DataFrameSelector(cat_atts)),
            ("imputer",MostFrequentImputer()),
            ("cat_encoder",OneHotEncoder(sparse=False))
])

cat = cat_pipeline.fit_transform(train_set)

pre_pipeline = FeatureUnion(transformer_list=[
                ("num_pipeline",num_pipeline),
                ("cat_pipeline",cat_pipeline),   
])

final = np.hstack((num,cat))
#print(final)

train = pre_pipeline.fit_transform(train_set)
y_train = train_set["Survived"]


#svm_clf = SVC(gamma="auto")
#svm_clf.fit(train,y_train)

forest_clf = RandomForestClassifier(n_estimators=150)
#forest_scores = cross_val_score(forest_clf,train,y_train,cv=10)

# test code
forest_clf.fit(train,y_train)
num_test = num_pipeline.fit_transform(test_set)
cat_test = cat_pipeline.fit_transform(test_set)
final_test = np.hstack((num_test,cat_test))

#test = pre_pipeline.fit(final_test)
#print(test)
y_test = forest_clf.predict(final_test)
#print(y_test)
passId = test_set["PassengerId"].astype('int32')
surv = y_test
ans = [passId,surv]
ans = list(map(list,zip(*ans)))
print(np.shape(ans))

with open('output.csv','w') as result:
    wr = csv.writer(result,dialect='excel')
    wr.writerows(ans)
"""
    My solution:
Pipeline to deal with the numerical attributes

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



