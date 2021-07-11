
import streamlit as st
import pickle
from PIL import Image
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import string
import nltk
import warnings
import pickle

image = Image.open('img.jpg')
model = pickle.load(open('model.pkl','rb'))
bow_vectorizer = pickle.load(open('bow_vectorizer.pkl','rb'))

def classify(num):
    if num[0][1]>0.4:
       st.success(" Tweet is positive")
    else: 
        st.error("Tweet is negative")

def remove_pattern(input_txt, pattern):
    r = re.findall(pattern, input_txt)
    for word in r:
        input_txt = re.sub(word, "", input_txt)
    return input_txt

def pre(text):
    text=[text]
    clean_tweet = np.vectorize(remove_pattern)(text, "@[\w]*")
    clean_tweet=pd.DataFrame(clean_tweet)
    clean_tweet[0] = clean_tweet[0].str.replace("[^a-zA-Z#]", " ")
    clean_tweet[0] = clean_tweet[0].apply(lambda x: " ".join([w for w in x.split() if len(w)>3]))
    tokenized_tweet = clean_tweet[0].apply(lambda x: x.split())
    from nltk.stem.porter import PorterStemmer
    stemmer = PorterStemmer()
    tokenized_tweet = tokenized_tweet.apply(lambda sentence: [stemmer.stem(word) for word in sentence])
    for i in range(len(tokenized_tweet)):
        tokenized_tweet[i] = " ".join(tokenized_tweet[i])
    clean_tweet[0] = tokenized_tweet
    bow = bow_vectorizer.transform(clean_tweet[0])
    pred_prob = model.predict_proba(bow)
    print(pred_prob)
    classify(pred_prob)

def main():
    st.title("Twitter Sentiment Analysis")
    st.write(" ")
    st.write(" ")
    st.write(" ")
    st.image(image)
    st.write(" ")
    st.markdown(
    """
    <style>
    .reportview-container {
        background: #aad8d3
    }
    </style>
    """, unsafe_allow_html=True)
   
    text = st.text_area("Enter the tweet",height = 200)
    st.write(" ")
    if st.button('Predict'):
        if text ==' ':
            st.warning("Please enetr some text")
        else:
            pre(text)


if __name__=='__main__':
    main()
