import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso


car_dataset = pd.read_csv('car data.csv')

# encoding "Fuel_Type" Column
car_dataset.replace({'Fuel_Type':{'Petrol':0,'Diesel':1,'CNG':2}},inplace=True)

# encoding "Seller_Type" Column
car_dataset.replace({'Seller_Type':{'Dealer':0,'Individual':1}},inplace=True)

# encoding "Transmission" Column
car_dataset.replace({'Transmission':{'Manual':0,'Automatic':1}},inplace=True)



import pickle
X = car_dataset.drop(['Car_Name','Selling_Price'],axis=1)
# print(X.info())
Y = car_dataset['Selling_Price']
lin_reg_model = LinearRegression()
x = lin_reg_model.fit(X,Y)
pickle.dump(x,open('lr.pkl','wb'))

lass_reg_model = Lasso()

x = lass_reg_model.fit(X,Y)
pickle.dump(x,open('la.pkl','wb'))