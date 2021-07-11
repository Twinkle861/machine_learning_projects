# Natural Language Processing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
# Importing the dataset
dataset = pd.read_csv('Restaurant_Reviews.tsv', delimiter = '\t', quoting = 3)

# Cleaning the texts
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus = []
for i in range(0, 1000):
  review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])
  review = review.lower()
  review = review.split()
  ps = PorterStemmer()
  all_stopwords = stopwords.words('english')
  all_stopwords.remove('not')
  all_stopwords.remove("isn't")
  review = [ps.stem(word) for word in review if not word in set(all_stopwords)]
  review = ' '.join(review)
  corpus.append(review)
# print(corpus)

# Creating the Bag of Words model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 1500)

cv1 = cv.fit(corpus)
print(type(cv1))
X = cv1.transform(corpus).toarray()
y = dataset.iloc[:, -1].values

# Splitting the dataset into the Training set and Test set
# from sklearn.model_selection import train_test_split
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

# Training the Naive Bayes model on the Training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
model = classifier.fit(X, y)

pickle.dump(model,open('model.pkl','wb'))
pickle.dump(cv1,open('cv1.pkl','wb'))


# text = "good"
# text=[text]

# text=pd.DataFrame(text)
# text[0] = text[0].str.replace("[^a-zA-Z]", " ")
# text[0] = text[0].str.lower()
# x = text[0].str.split()

# stemmer = PorterStemmer()
# all_stopwords = stopwords.words('english')
# all_stopwords.remove('not')
# all_stopwords.remove("isn't")
# x= x.apply(lambda sentence: [stemmer.stem(word) for word in sentence if not word in set(all_stopwords)])
# for i in range(len(x)):
#     x[i] = " ".join(x[i])
# print(x)
# text[0] = x
# cv12 = cv1.transform(text[0]).toarray()
# pred_prob = model.predict_proba(cv12)
# print(pred_prob)
# num = pred_prob
# if num[0][0]>0.5:
#     print("Review is negative ")
# else:
#     print("Review is positive")