import streamlit as st
import pickle
from PIL import Image
import time 
import numpy 
from tensorflow.keras.models import load_model
import pickle
image = Image.open('img.png')

rnn = load_model('model.h5')
sc = pickle.load(open("sc.pkl","rb"))
def print1(num):
    if float(num[0][0])>0.7:
        st.error('Customer leaves the bank.😰 ')
    else:
        st.success('Customer stays in the bank.🥰 ')

def main():
    st.title("Customer stay/leave Predictor")
    st.image(image)
    st.markdown(
    """
    <style>
    .reportview-container {
        background: #f6dfeb
    }
    </style>
    """, unsafe_allow_html=True)
    
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    
    st.text_input("Enter CustomerID")
    st.text_input("Enter surname")

    geography = st.selectbox("Enter Geography",["France","Spain","Germany"])
    le.classes_ = numpy.load('geography.npy',allow_pickle=True)
    geography = le.transform([geography])

    gender = st.selectbox("Enter Gender",["Male","Female"])
    le.classes_ = numpy.load('gender.npy',allow_pickle=True)
    gender = le.transform([gender])

    cscore = st.number_input("Enter Credit Card Score",min_value =1.00)
    age = st.number_input("Enter Age of Customer",min_value =1)
    tenure = st.number_input("Enter tenure",min_value =1.00)
    balance = st.number_input("Enter Balance",min_value =0.00)
    nproducts = st.number_input("Enter Number of products customer has",min_value =1)
    card = st.checkbox("Has credit card")
    active = st.checkbox("Is an acive member")
    salary = st.number_input("Enter estimated salary",min_value =1.00)


    inputs = sc.transform([[cscore,geography[0],gender[0],age,tenure,balance,nproducts,int(card),int(active),salary]])

    if st.button('Predict'):
            print1(rnn.predict_proba(inputs))


if __name__=='__main__':
    main()
