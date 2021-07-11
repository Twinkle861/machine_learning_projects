import pandas as pd
import numpy as np
import pickle
df = pd.read_csv('titanic.csv')
df.drop("Cabin", axis = 1,inplace = True)
df.dropna(inplace =True)
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df['Embarked']=le.fit_transform(df['Embarked'].values)
np.save('embarked.npy', le.classes_)
# print(le.classes_)
df['Sex']=le.fit_transform(df['Sex'].values)
np.save('sex.npy', le.classes_)
# print(le.classes_)


df.drop(['PassengerId','Name','Ticket'],axis = 1,inplace = True)
x = df.drop("Survived",axis = 1)

y= df["Survived"]
# print(x.info())

from sklearn.preprocessing import StandardScaler
sc =StandardScaler()
x = sc.fit_transform(x)
pickle.dump(sc,open('sc.pkl','wb'))

from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
model = lr.fit(x,y)

pickle.dump(model,open('lr.pkl','wb'))