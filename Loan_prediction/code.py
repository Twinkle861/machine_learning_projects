import pandas as pd
import numpy as np
import pickle

df = pd.read_csv("Loan Prediction Dataset.csv")

# drop unnecessary columns
cols = ['ApplicantIncome', 'CoapplicantIncome', "LoanAmount", "Loan_Amount_Term", 'Loan_ID']
df = df.drop(columns=cols, axis=1)

# fill the missing values for numerical terms - mean
df['Credit_History'] = df['Credit_History'].fillna(df['Credit_History'].mean())

# fill the missing values for categorical terms - mode
df['Gender'] = df["Gender"].fillna(df['Gender'].mode()[0])
df['Married'] = df["Married"].fillna(df['Married'].mode()[0])
df['Dependents'] = df["Dependents"].fillna(df['Dependents'].mode()[0])
df['Self_Employed'] = df["Self_Employed"].fillna(df['Self_Employed'].mode()[0])

from sklearn.preprocessing import LabelEncoder
cols = ['Gender',"Married","Education",'Self_Employed',"Property_Area","Loan_Status","Dependents"]
le = LabelEncoder()
for col in cols:
    df[col] = le.fit_transform(df[col])

# specify input and output attributes
X = df.drop(columns=['Loan_Status'], axis=1)
y = df['Loan_Status']

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier,ExtraTreesClassifier

log_model = LogisticRegression()
dec_model = DecisionTreeClassifier()
extree_model = ExtraTreesClassifier()
rantree_model = RandomForestClassifier(n_estimators=100, min_samples_split=25, max_depth=7, max_features=1)

log_model = log_model.fit(x_train,y_train)
dec_model = dec_model.fit(x_train,y_train)
extree_model = extree_model.fit(x_train,y_train)
rantree_model = rantree_model.fit(x_train,y_train)

pickle.dump(le,open('label_encoder.pkl','wb'))
pickle.dump(log_model,open('log_model.pkl','wb'))
pickle.dump(dec_model,open('dec_model.pkl','wb'))
pickle.dump(extree_model,open('extree_model.pkl','wb'))
pickle.dump(rantree_model,open('rantree_model.pkl','wb'))

print(x_train.describe())