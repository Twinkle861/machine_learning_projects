from nltk.corpus.reader.pl196x import TYPE
import streamlit as st
import pickle
from PIL import Image


image = Image.open('img.jpg')

li_model = pickle.load(open('li_model.pkl','rb'))
nav_model = pickle.load(open('nav_model.pkl','rb'))
ran_model = pickle.load(open('ran_model.pkl','rb'))
svc_model = pickle.load(open('svc_model.pkl','rb'))


def clean(text):
    import nltk
    import re
    nltk.download('stopwords')
    nltk.download('punkt')
    from nltk.corpus import stopwords
    STOPWORDS = set(stopwords.words('english'))
    # convert to lowercase
    text1 = text.lower()
    # remove special characters
    text1 = re.sub(r'[^0-9a-zA-Z]', ' ', text1)
    # remove extra spaces
    text1 = re.sub(r'\s+', ' ', text1)
    # remove stopwords
    text1 = " ".join(word for word in text1.split() if word not in STOPWORDS)
    return text1



def classify(num):
    if num == 'spam':
        st.error('Message is predicted as a Spam')
    else:
        st.success('Message is predicted as not a Spam')

def main():
    st.title("Spam /Not Spam Predictor")
    st.write(" ")
    st.write(" ")
    st.write(" ")
    st.image(image)
    st.write(" ")
    st.write(" ")
    st.markdown(
    """
    <style>
    .reportview-container {
        background: #94d0cc
    }
    </style>
    """, unsafe_allow_html=True)
    activities=['Logistic Regression','Naive Bayes', 'Suppot Vector Machine', 'Random Forest']
    option=st.sidebar.selectbox('Which model would you like to use?',activities)
    st.subheader(option)

    text = st.text_area("Enter the message")

    

    if st.button('Predict'):
        text = clean(text)
        inputs = [text]
        if option=='Support Vector Machine':
            classify(svc_model.predict(inputs))
        elif option=='Logistic Regression':
            classify(li_model.predict(inputs))
        elif option=='Naive Bayes':
            classify(nav_model.predict(inputs))
        else:
           classify(ran_model.predict(inputs))


if __name__=='__main__':
    main()
