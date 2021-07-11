
# hyperparameter-Gridsearch(use all combinations-most time but best value),randomsearch(take values randomly,get close to best result)
# bayesiansearch-best technique->
# install module


import pandas as pd
import numpy as np
import pickle
# import seaborn as sns
# import matplotlib.pyplot as plt

df = pd.read_csv('Train.csv')

item_weight_mean = df.pivot_table(values = "Item_Weight", index = 'Item_Identifier')
miss_bool = df['Item_Weight'].isnull()
for i, item in enumerate(df['Item_Identifier']):
    if miss_bool[i]:
        if item in item_weight_mean:
            df['Item_Weight'][i] = item_weight_mean.loc[item]['Item_Weight']
        else:
            df['Item_Weight'][i] = np.mean(df['Item_Weight'])

outlet_size_mode = df.pivot_table(values='Outlet_Size', columns='Outlet_Type', aggfunc=(lambda x: x.mode()[0]))
miss_bool = df['Outlet_Size'].isnull()
df.loc[miss_bool, 'Outlet_Size'] = df.loc[miss_bool, 'Outlet_Type'].apply(lambda x: outlet_size_mode[x])

# replace zeros with mean
df.loc[:, 'Item_Visibility'].replace([0], [df['Item_Visibility'].mean()], inplace=True)

df['Item_Fat_Content'] = df['Item_Fat_Content'].replace({'LF':'Low Fat', 'reg':'Regular', 'low fat':'Low Fat'})
df['New_Item_Type'] = df['Item_Identifier'].apply(lambda x: x[:2])
df['New_Item_Type'] = df['New_Item_Type'].map({'FD':'Food', 'NC':'Non-Consumable', 'DR':'Drinks'})
df.loc[df['New_Item_Type']=='Non-Consumable', 'Item_Fat_Content'] = 'Non-Edible'

df['Outlet_Years'] = 2021 - df['Outlet_Establishment_Year']


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df['Outlet'] = le.fit_transform(df['Outlet_Identifier'])
print(le.classes_)
np.save('Outlet.npy', le.classes_)
cat_col = ['Item_Fat_Content', 'Item_Type', 'Outlet_Size', 'Outlet_Location_Type', 'Outlet_Type', 'New_Item_Type']
for col in cat_col:
    df[col] = le.fit_transform(df[col])
    name = col + '.npy'
    # print(le.classes_)
    np.save(name, le.classes_)

X = df.drop(columns=['Outlet_Establishment_Year', 'Item_Identifier', 'Outlet_Identifier','Item_Outlet_Sales'])
y = df['Item_Outlet_Sales']
# print(X.head())
# print(y.head())
# print(X.info())

from sklearn.linear_model import LinearRegression, Ridge, Lasso
model = LinearRegression(normalize=True)
lr = model.fit(X, y)

model = Ridge(normalize=True)
ri = model.fit(X, y)

model = Lasso()
la = model.fit(X, y)

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
pickle.dump(ri,open('ri.pkl','wb'))
pickle.dump(la,open('la.pkl','wb'))
pickle.dump(dt,open('dt.pkl','wb'))
pickle.dump(et,open('et.pkl','wb'))
pickle.dump(rf,open('rf.pkl','wb'))