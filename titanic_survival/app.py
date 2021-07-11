import streamlit as st
import pickle
from PIL import Image
import time 
import numpy 

image = Image.open('img.jpg')

lr = pickle.load(open('lr.pkl','rb'))
sc = pickle.load(open('sc.pkl','rb'))


def print1(num):
    if num==0:
         st.error("Sorry...You wouldn't have survived.🥺")
    else:
        st.success("Congratulations...You would have survived.🙂✨")

def main():
    st.title("Titanic Survival Predictor")
    st.image(image)
    st.markdown(
    """
    <style>
    .reportview-container {
        background: #5F939A
    }
    </style>
    """, unsafe_allow_html=True)
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()

    st.text_input("Enter passenger id")
    pclass = st.number_input("Enter passenger class",min_value=1)
    st.text_input("Enter name of passenger")
    sex = st.selectbox("Enter gender",['male','female'])
    le.classes_ = numpy.load('sex.npy',allow_pickle=True)
    sex = le.transform([sex])
    age = st.number_input("Enter age of passenger")
    sib = st.number_input("Enter number of siblings or spouse",min_value=0)
    par = st.number_input("Enter number of parents or children",min_value =0)
    st.text_input("Enter ticket number")
    fare = st.number_input("Enter fare",min_value=0.0)
    st.text_input("Enter cabin")
    embarked = st.selectbox("Where do you belong",['C','Q','S'])
    le.classes_ = numpy.load('embarked.npy',allow_pickle=True)
    embarked = le.transform([embarked])


    inputs = [[pclass,sex[0],age,sib,par,fare,embarked[0]]]
    inputs = sc.transform(inputs)

    if st.button('Predict'):
            print1(lr.predict(inputs))
        


if __name__=='__main__':
    main()
