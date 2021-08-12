
# #black friday sales
# mask->to hide info not avaible to public
# predict purchase amount
# bivariate analysis=>analysis two independent variables wrt to each other
# encode using dict or label encoding

import pandas as pd
import numpy as np
import pickle 

df = pd.read_csv('train.csv')

df['Product_Category_2'] = df['Product_Category_2'].fillna(-2.0).astype("float32")
df['Product_Category_3'] = df['Product_Category_3'].fillna(-2.0).astype("float32")

cols = ['Gender','Age', 'City_Category', 'Stay_In_Current_City_Years']
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
for col in cols:
    df[col] = le.fit_transform(df[col])
    name = col + '.npy'
    # print(le.classes_)
    np.save(name, le.classes_)

X = df.drop(columns=['User_ID', 'Product_ID', 'Purchase'])
y = df['Purchase']
# print(X.info())

from sklearn.linear_model import LinearRegression
model = LinearRegression(normalize=True)
lr = model.fit(X, y)

from sklearn.tree import DecisionTreeRegressor
model = DecisionTreeRegressor()
dt = model.fit(X, y)

from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor()
rf = model.fit(X, y)

from sklearn.ensemble import ExtraTreesRegressor
model = ExtraTreesRegressor()
et = model.fit(X, y)

pickle.dump(lr,open('lr.pkl','wb'))
pickle.dump(dt,open('dt.pkl','wb'))
pickle.dump(et,open('et.pkl','wb'))
pickle.dump(rf,open('rf.pkl','wb'))
