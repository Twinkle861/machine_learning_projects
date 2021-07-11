# #bike sharing
# max_columns to see all features
# use correlation only for int variables
# cv score ->error metric therfore les the better

import pandas as pd
import numpy as np
import pickle

df = pd.read_csv('hour.csv')


df = df.rename(columns={'weathersit':'weather',
                       'yr':'year',
                       'mnth':'month',
                       'hr':'hour',
                       'hum':'humidity',
                       'cnt':'count'})

df = df.drop(columns=['instant', 'dteday', 'year'])

X = df.drop(columns=['atemp', 'windspeed', 'casual', 'registered', 'count'], axis=1)
# X.info()
y = df['count']

from sklearn.linear_model import LinearRegression, Ridge, HuberRegressor, ElasticNetCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor

model = LinearRegression()
lr = model.fit(X, y)

model = Ridge()
ri = model.fit(X, y)

model = HuberRegressor()
hr = model.fit(X, y)

model = ElasticNetCV()
en = model.fit(X, y)

model = DecisionTreeRegressor()
dt = model.fit(X, y)

model = GradientBoostingRegressor()
gb = model.fit(X, y)

model = ExtraTreesRegressor()
et = model.fit(X, y)

model = RandomForestRegressor()
rf = model.fit(X, y)


pickle.dump(lr,open('lr.pkl','wb'))
pickle.dump(gb,open('gb.pkl','wb'))
pickle.dump(ri,open('ri.pkl','wb'))
pickle.dump(hr,open('hr.pkl','wb'))
pickle.dump(en,open('en.pkl','wb'))
pickle.dump(dt,open('dt.pkl','wb'))
pickle.dump(et,open('et.pkl','wb'))
pickle.dump(rf,open('rf.pkl','wb'))