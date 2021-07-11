
# #wine quality prediction->
# minimeise error=>reg
# get max accuracy=>classification
# all classes are not balanced therefore need to balance them ie almost equal values for quality
# take smode=>avg features from neigbours and create new class
# make all class equal
# use class balncement t improve accuracy
# also can remove outliers and droop col on basis of corr matrix(density and sulphur dioxode),normalise the data,random undersmaplind and over sampling

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
import pickle

df = pd.read_csv('winequality.csv')


# fill the missing values
for col, value in df.items():
    if col != 'type':
        df[col] = df[col].fillna(df[col].mean())


X = df.drop(columns=['type', 'quality','free sulfur dioxide'])
y = df['quality']

from imblearn.over_sampling import SMOTE
oversample = SMOTE(k_neighbors=4)
# transform the dataset
X, y = oversample.fit_resample(X, y)

# classify function
from sklearn.model_selection import cross_val_score, train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
   

from sklearn.ensemble import ExtraTreesClassifier
extra_model = ExtraTreesClassifier()
# train the model
extra_model=extra_model.fit(x_train, y_train)
# print("Accuracy:", model.score(x_test, y_test) * 100)  
#  # cross-validation
# score = cross_val_score(model, X, y, cv=5)
# print("CV Score:", np.mean(score)*100)

import lightgbm 
light_model = lightgbm.LGBMClassifier()
light_model=light_model.fit(x_train, y_train)

from sklearn.ensemble import RandomForestClassifier
random_model = RandomForestClassifier()
random_model=random_model.fit(x_train, y_train)


from sklearn.tree import DecisionTreeClassifier
decision_model = DecisionTreeClassifier()
decision_model=decision_model.fit(x_train, y_train)


pickle.dump(extra_model,open('extra_model.pkl','wb'))
pickle.dump(light_model,open('light_model.pkl','wb'))
pickle.dump(decision_model,open('decision_model.pkl','wb'))
pickle.dump(random_model,open('random_model.pkl','wb'))
