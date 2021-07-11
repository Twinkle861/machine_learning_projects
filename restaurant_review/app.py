
import streamlit as st
import pickle
from PIL import Image
import pandas as pd
import numpy as np
# nltk.download('stopwords')
from nltk.corpus import stopwords
import nltk
import pickle

image = Image.open('img.jpg')
model = pickle.load(open('model.pkl','rb'))
cv = pickle.load(open('cv1.pkl','rb'))

def classify(num):
    if num[0][0]>0.5:
       st.error("Review is negative 😓")
    else: 
        st.success("Review is positive 😍 ")

def pre(text):
    text=[text]
    text=pd.DataFrame(text)
    text[0] = text[0].str.replace("[^a-zA-Z]", " ")
    text[0] = text[0].str.lower()
    x = text[0].str.split()

    from nltk.stem.porter import PorterStemmer
    stemmer = PorterStemmer()
    all_stopwords = stopwords.words('english')
    all_stopwords.remove('not')
    all_stopwords.remove("isn't")
    x= x.apply(lambda sentence: [stemmer.stem(word) for word in sentence if not word in set(all_stopwords)])
    for i in range(len(x)):
        x[i] = " ".join(x[i])
    # print(x)
    text[0] = x
    # print(text[0][0])
    cv1 = cv.transform(text[0]).toarray()
    pred_prob = model.predict_proba(cv1)
    print(pred_prob)
    classify(pred_prob)

def main():
    st.title("Restaurant review Analysis")
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
   
    text = st.text_area("Enter the review by customer",height = 200)
    st.write(" ")
    if st.button('Predict'):
        if text ==' ':
            st.warning("Please enetr some text")
        else:
            pre(text)


if __name__=='__main__':
    main()
