# #boston house prediction
# 1 coressponds to $1000
# predict price
# have outliers
# do log transformation or drop that columns or drop outliers row
# do min max normalisation on skewd and double bell one  get ranges 0 to 1=>do for any column ie is not uniform distributed
# standardisation to get uniform distribution
# ignore 1 in highly correlated =>can ignore lat
# -vely corelated when one is dirctly proporitional to dependent and other is inversely proportional
# cv = k cross fold validation
# both mse and cv score must be less=>better the model as comparing error
# use xgboost
import pandas as pd
import numpy as np
import pickle


df = pd.read_csv("Boston Dataset.csv")
df.drop(columns=['Unnamed: 0'], axis=0, inplace=True)

cols = ['crim', 'zn', 'tax', 'black']
for col in cols:
    # find minimum and maximum of that column
    minimum = min(df[col])
    maximum = max(df[col])
    df[col] = (df[col] - minimum) / (maximum - minimum)
    arr = [minimum,maximum]
    name = col +'.pkl'
    pickle.dump(arr,open(name,'wb'))

from sklearn import preprocessing
scalar = preprocessing.StandardScaler()
sc = scalar.fit(df[cols])
pickle.dump(scalar,open('sc.pkl','wb'))
# fit our data
scaled_cols = sc.transform(df[cols])
scaled_cols = pd.DataFrame(scaled_cols, columns=cols)
for col in cols:
    df[col] = scaled_cols[col]


X = df.drop(columns=['medv', 'rad'], axis=1)
y = df['medv']



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

# import xgboost as xgb
# model = xgb.XGBRegressor()
# xg = model.fit(X, y)

pickle.dump(lr,open('lr.pkl','wb'))
pickle.dump(dt,open('dt.pkl','wb'))
pickle.dump(et,open('et.pkl','wb'))
pickle.dump(rf,open('rf.pkl','wb'))
# pickle.dump(xg,open('xg.pkl','wb'))
