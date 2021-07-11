import pandas as pd
import numpy as np
import pickle

data = pd.read_csv('data.csv')

data.dropna(how='any',inplace=True)
data= data.loc[(data.strength == '1') | (data.strength == '2') | (data.strength == '0'),:]
passwords_tuple=np.array(data)
import random
random.shuffle(passwords_tuple)


X=[labels[0] for labels in passwords_tuple]
y=[labels[1] for labels in passwords_tuple]

def char_tokenize(inputs):
    characters=[]
    for i in inputs:
        characters.append(i)
    return characters


from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer=TfidfVectorizer()
tr=vectorizer.fit(X)
pickle.dump(tr,open('tr.pkl','wb'))
X=tr.transform(X)

# Logistic Regression

from sklearn.linear_model import LogisticRegression
log_class=LogisticRegression(penalty='l2',multi_class='ovr')
lg = log_class.fit(X,y)
pickle.dump(lg,open('lg.pkl','wb'))

from sklearn.naive_bayes import BernoulliNB
BNB = BernoulliNB()
bn = BNB.fit(X,y)
pickle.dump(bn,open('bn.pkl','wb'))

from sklearn.tree import DecisionTreeClassifier
dtc = DecisionTreeClassifier()
dt = dtc.fit(X,y)
pickle.dump(dt,open('dt.pkl','wb'))

from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=100, max_depth=50, criterion='entropy')
rf = rfc.fit(X,y)
pickle.dump(rf,open('rf.pkl','wb'))

from sklearn.neural_network import MLPClassifier
mlp = MLPClassifier(solver='adam', alpha=1e-5, max_iter=400, activation='logistic')
ml = mlp.fit(X,y)
pickle.dump(ml,open('ml.pkl','wb'))
