import streamlit as st
import pickle
from PIL import Image
import time 
import numpy 

image = Image.open('img.jpg')

lr = pickle.load(open('lr.pkl','rb'))
dt = pickle.load(open('dt.pkl','rb'))
et = pickle.load(open('et.pkl','rb'))
rf = pickle.load(open('rf.pkl','rb'))

sc = pickle.load(open('sc.pkl','rb'))

def print1(num):
    num=num[0]
    return "Median value of owner-occupied homes in $1000's is -> " + str(num)

def main():
    st.title("Boston Housing Predictor")
    st.image(image)
    st.markdown(
    """
    <style>
    .reportview-container {
        background: #f6dfeb
    }
    </style>
    """, unsafe_allow_html=True)
    activities=['LinearRegression','Decision Tree', 'Extra Tree Regressor', 'Random Forest']
    option=st.sidebar.selectbox('Which model would you like to use?',activities)
    st.subheader(option)
    

    cr = st.number_input("Enter per capita crime rate by town")

    arr = pickle.load(open('crim.pkl','rb'))
    min = arr[0]
    max = arr[1]
    cr = (cr - min) / (max - min)
    

    zn = st.number_input("Enter proportion of residential land zoned for lots over 25,000 sq.ft.")

    arr = pickle.load(open('zn.pkl','rb'))
    min = arr[0]
    max = arr[1]
    zn = (zn - min) / (max - min)

    ind = st.number_input("Enter proportion of non-retail business acres per town")   
    ch = st.selectbox("Select 1 if tract bounds river else 0",('1', '0'))
    ch = int(ch)
    nox = st.number_input("Enter nitric oxides concentration (parts per 10 million).")
    rm = st.number_input("Enter average number of rooms per dwelling ")
    age = st.number_input("Enter proportion of owner-occupied units built prior to 1940")
    dis = st.number_input("Enter weighted distances to five Boston employment centres")
    st.number_input("Enter index of accessibility to radial highways")
    tax = st.number_input("Enter full-value property-tax rate per $10,000")

    arr = pickle.load(open('tax.pkl','rb'))
    min = arr[0]
    max = arr[1]
    tax = (tax - min) / (max - min)
  

    p = st.number_input("Enter pupil-teacher ratio by town")
    b = st.number_input("Enter 1000(poprtion of black in town - 0.63)^2")

    arr = pickle.load(open('black.pkl','rb'))
    min = arr[0]
    max = arr[1]
    b = (b - min) / (max - min)
  

    ls = st.number_input("Enter lower status of the population")
    
    x = sc.transform([[cr,zn,tax,b]])
    cr = x[0][0]
    zn = x[0][1]
    tax = x[0][2]
    b = x[0][3]

    inputs = [[cr, zn, ind ,ch, nox, rm, age,dis, tax, p, b, ls ]]

    if st.button('Predict'):
        if option=='Decision Tree':
            st.success(print1(dt.predict(inputs)))
        elif option=='LinearRegression':
            st.success(print1(lr.predict(inputs)))
        elif option=='Extra Tree Regressor':
            st.success(print1(et.predict(inputs)))
        else:
            st.success(print1(rf.predict(inputs)))


if __name__=='__main__':
    main()
