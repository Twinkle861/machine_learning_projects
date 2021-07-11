#spam
# to apply to each row=>use.apply(func nmae)
# stratify for equal distribution of classes
import pandas as pd
import numpy as np
import nltk
import re
nltk.download('stopwords')
nltk.download('punkt')
from nltk.corpus import stopwords
import pickle

df = pd.read_csv('spam.csv')
# get necessary columns for processing
df = df[['v2', 'v1']]
# df.rename(columns={'v2': 'messages', 'v1': 'label'}, inplace=True)
df = df.rename(columns={'v2': 'messages', 'v1': 'label'})


STOPWORDS = set(stopwords.words('english'))

def clean_text(text):
    # print(type(text))
    # convert to lowercase
    text = text.lower()
    # remove special characters
    text = re.sub(r'[^0-9a-zA-Z]', ' ', text)
    # remove extra spaces
    text = re.sub(r'\s+', ' ', text)
    # remove stopwords
    text = " ".join(word for word in text.split() if word not in STOPWORDS)
    return text

df['clean_text'] = df['messages'].apply(clean_text)

X = df['clean_text']
y = df['label']

from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer

def classify(model, X, y):
#     print(model)
    if model == li_model:
       name = 'li_model'
    elif model == nav_model:
        name = 'nav_model'
    elif model == svc_model:
        name = 'svc_model'
    else:
        name = 'ran_model'

    # train test split
    # x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, shuffle=True, stratify=y)
    # model training
    pipeline_model = Pipeline([('vect', CountVectorizer()),
                              ('tfidf', TfidfTransformer()),
                              ('clf', model)])
    pipeline_model.fit(X, y)
    path = name + '.pkl'
    pickle.dump(pipeline_model,open(path,'wb'))
    
    # print('Accuracy:', pipeline_model.score(x_test, y_test)*100)
    # print(pipeline_model.predict(x_test))
    
from sklearn.linear_model import LogisticRegression
li_model = LogisticRegression()
classify(li_model,X,y)

from sklearn.naive_bayes import MultinomialNB
nav_model = MultinomialNB()
classify(nav_model,X,y)

from sklearn.svm import SVC
svc_model = SVC(C=3)
classify(svc_model,X,y)

from sklearn.ensemble import RandomForestClassifier
ran_model = RandomForestClassifier()
classify(ran_model,X,y)

