import streamlit as st
import pickle
from PIL import Image
import time 
import numpy 

image = Image.open('img.jpeg')

lr = pickle.load(open('lr.pkl','rb'))
la = pickle.load(open('la.pkl','rb'))


def print1(num):
    num=num[0]
    if int(num)<0:
         num=0
    return 'Selling Price prediction is ' + str(num)

def main():
    st.title("Used Car Price Predictor")
    st.image(image)
    st.markdown(
    """
    <style>
    .reportview-container {
        background: #f6dfeb
    }
    </style>
    """, unsafe_allow_html=True)
    activities=['LinearRegression', 'Lasso']
    option=st.sidebar.selectbox('Which model would you like to use?',activities)
    st.subheader(option)
    
    st.text_input("Enter car name")
    year = st.number_input("Enter the model's year",min_value=1900)
    price = st.number_input("Enter the present price",min_value=1.0)
    km = st.number_input("Enter the number of kilometers driven",min_value=0.0)
    fuel = st.selectbox("Enter the fuel type",['Petrol','Diesel','CNG'])
    if fuel == 'Petrol':
        fuel = 0
        fuel = int(fuel)
    elif fuel == 'Diesel':
        fuel = 1
        fuel = int(fuel)
    else:
        fuel = 2
        fuel = int(fuel)
    seller = st.selectbox("Enter the seller type",['Dealer','Individual'])
    if seller == 'Dealer':
        seller = 0
        seller = int(seller)
    else:
        seller = 1
        seller = int(seller)
    trans = st.selectbox("Enter the gear type",['Manual','Automatic'])
    if trans == 'Manual':
        trans = 0
        trans = int(trans)
    else:
        trans = 1
        trans = int(trans)
        
    owner = st.number_input("Enter number of owners before you",min_value=0)


    inputs = [[year,price,km,fuel,seller,trans,owner]]

    if st.button('Predict'):
        if option=='LinearRegression':
            st.success(print1(lr.predict(inputs)))
        elif option=='Lasso':
            st.success(print1(la.predict(inputs)))
        


if __name__=='__main__':
    main()
