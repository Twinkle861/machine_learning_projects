import streamlit as st
import pickle
from PIL import Image
import time 
import numpy 

image = Image.open('img.jpg')


lr = pickle.load(open('lr.pkl','rb'))
gb = pickle.load(open('gb.pkl','rb'))
ri = pickle.load(open('ri.pkl','rb'))
hr = pickle.load(open('hr.pkl','rb'))
en = pickle.load(open('en.pkl','rb'))
dt = pickle.load(open('dt.pkl','rb'))
et = pickle.load(open('et.pkl','rb'))
rf = pickle.load(open('rf.pkl','rb'))


def print1(num):
    st.write(" ")
    num=num[0]
    return "Count of total rental bikes including both casual and registered is -> " + str(num)

def main():
    st.title("Count of Total Rental Bikes Predictor")
    st.write(" ")
    st.write(" ")
    st.image(image)
    st.markdown(
    """
    <style>
    .reportview-container {
        background: #f6dfeb
    }
    </style>
    """, unsafe_allow_html=True)
    activities=['LinearRegression','Decision Tree', 'Extra Tree Regressor', 'Random Forest','HuberRegressor','ElasticNetCV','GradientBoostingRegressor','Ridge']
    st.write(" ")

    option=st.sidebar.selectbox('Which model would you like to use?',activities)
    st.subheader(option)
    st.write(" ")
    st.write(" ")

    st.number_input("Enter record instant")
    st.write(" ")

    st.date_input("Enetr date")
    st.write(" ")
    
    season = st.selectbox("Select the season",["Winter","Spring","Summer","fall"])
    if season == "Winter":
        season = 1
    elif season == "Spring":
        season =2
    elif season == "Summer":
        season = 3
    else:
        season =4
    st.write(" ")
    
    st.selectbox("Enter year",["2011","2012"])
    st.write(" ")
    
    mnth = st.number_input("Enter Month(from 1 to 12)",min_value=1,max_value=12)
    st.write(" ")
    hr = st.number_input("Enter hour(24-hr format)",min_value=0,max_value=23)
    st.write(" ")
    holiday = st.checkbox("It is a holiday")
    if holiday == False:
        holiday = 0
    else:
        holiday = 1
    st.write(" ")
    weekday = st.number_input("Enter weekday number(from 0 t0 6)",min_value =0 ,max_value=6)
    st.write(" ")
    workingday = st.checkbox("It is a working day(ie. neither weeknd nor holiday")
    if workingday == False:
        workingday = 0
    else:
        workingday = 1
    st.write(" ")
    weather = st.selectbox("Enter weather condition",["Clear, Few clouds, Partly cloudy, Partly cloudy","Mist + Cloudy, Mist + Broken clouds, Mist + Few clouds, Mist","Light Snow, Light Rain + Thunderstorm + Scattered clouds, Light Rain + Scattered clouds","Heavy Rain + Ice Pallets + Thunderstorm + Mist, Snow + Fog"])
    if weather == "Clear, Few clouds, Partly cloudy, Partly cloudy":
        weather = 1
    elif weather == "Mist + Cloudy, Mist + Broken clouds, Mist + Few clouds, Mist":
        weather =2
    elif weather == "Light Snow, Light Rain + Thunderstorm + Scattered clouds, Light Rain + Scattered clouds":
        weather = 3
    else:
        weather =4
    st.write(" ")
    temp = st.number_input("Enter temperature in celsius",min_value=0.02,max_value=1.00)
    st.write(" ")
    st.number_input("Enter normalised feeling temperature in celsius",min_value=0.0,max_value=1.00)
    st.write(" ")
    hum = st.number_input("Enter humidity",min_value=0.0,max_value=1.00)
    st.write(" ")
    st.number_input("Enter windspeed",min_value=0.0,max_value=0.86)
    st.write(" ")
    st.number_input("Enter number of casual users")
    st.write(" ")
    st.number_input("Enter count of registered users")

    inputs = [[season, mnth, hr, holiday, weekday, workingday,weather,temp,hum]]
    st.write(" ")
    st.write(" ")

    if st.button('Predict'):
        if option=='Decision Tree':
            st.success(print1(dt.predict(inputs)))
        elif option=='LinearRegression':
            st.success(print1(lr.predict(inputs)))
        elif option=='Extra Tree Regressor':
            st.success(print1(et.predict(inputs)))
        elif option=='HuberRegressor':
            st.success(print1(et.predict(inputs)))
        elif option=='ElasticNetCV':
            st.success(print1(et.predict(inputs)))
        elif option=='GradientBoostingRegressor':
            st.success(print1(et.predict(inputs)))
        elif option=='Ridge':
            st.success(print1(et.predict(inputs)))
        else:
            st.success(print1(rf.predict(inputs)))


if __name__=='__main__':
    main()
