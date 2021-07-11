import numpy as np
import pandas as pd
import pickle

dataset = pd.read_csv('weather.csv')
X = dataset.iloc[:,[1,2,3,4,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21]].values
Y = dataset.iloc[:,-1].values

Y = Y.reshape(-1,1)

from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan,strategy='most_frequent')
X = imputer.fit_transform(X)
Y = imputer.fit_transform(Y)

from sklearn.preprocessing import LabelEncoder
le1 = LabelEncoder()
X[:,0] = le1.fit_transform(X[:,0])
np.save('1.npy', le1.classes_)

le2 = LabelEncoder()
X[:,4] = le2.fit_transform(X[:,4])
np.save('2.npy', le2.classes_)

le3 = LabelEncoder()
X[:,6] = le3.fit_transform(X[:,6])
np.save('3.npy', le3.classes_)

le4 = LabelEncoder()
X[:,7] = le4.fit_transform(X[:,7])
np.save('4.npy', le4.classes_)
le5 = LabelEncoder()
X[:,-1] = le5.fit_transform(X[:,-1])
np.save('5.npy', le5.classes_)
le6 = LabelEncoder()
Y[:,-1] = le6.fit_transform(Y[:,-1])

# print(le1.classes_)
# print(le2.classes_)
# print(le3.classes_)
# print(le4.classes_)
# print(le5.classes_)
# print(le6.classes_)
Y = np.array(Y,dtype=float)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
sc = sc.fit(X)
X = sc.transform(X)
pickle.dump(sc,open('sc.pkl','wb'))

from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators=100,random_state=0)
rf = classifier.fit(X,Y)
pickle.dump(rf,open('rf.pkl','wb'))
# print(X.descibe())