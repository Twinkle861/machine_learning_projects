import streamlit as st
import pickle
from PIL import Image
import time 
import numpy 

image = Image.open('img.jpeg')

rf = pickle.load(open('rf.pkl','rb'))
sc = pickle.load(open('sc.pkl','rb'))

def print1(num):
    num=num[0]
    if int(num) == 1:
        st.success("It will rain tomorrow.")
    else:
        st.error("It wont rain tomorrow.")

def main():
    st.title("Rainfall Predictor")
    st.write(" ")
    st.write(" ")
    st.image(image)
    st.write(" ")

    st.markdown(
    """
    <style>
    .reportview-container {
        background: #f6dfeb
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.subheader("Enter the following details:")
    st.write(" ")

    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()

    st.date_input("Enter the date")

    loc = st.selectbox("Enter the location",['Adelaide', 'Albany', 'Albury', 'AliceSprings','BadgerysCreek', 'Ballarat', 'Bendigo', 'Brisbane', 'Cairns', 'Canberra', 'Cobar', 'CoffsHarbour', 'Dartmoor', 'Darwin', 'GoldCoast', 'Hobart', 'Katherine', 'Launceston', 'Melbourne', 'MelbourneAirport', 'Mildura', 'Moree', 'MountGambier', 'MountGinini', 'Newcastle', 'Nhil', 'NorahHead', 'NorfolkIsland', 'Nuriootpa','PearceRAAF', 'Penrith', 'Perth', 'PerthAirport', 'Portland', 'Richmond', 'Sale', 'SalmonGums', 'Sydney', 'SydneyAirport', 'Townsville', 'Tuggeranong', 'Uluru', 'WaggaWagga', 'Walpole', 'Watsonia', 'Williamtown', 'Witchcliffe', 'Wollongong', 'Woomera'])
    le.classes_ = numpy.load('1.npy',allow_pickle=True)
    loc = le.transform([loc])

    mint = st.number_input("Enter the minimum temperature in degrees celsius")
    maxt = st.number_input("Enter the maximum temperature in degrees celsius")
    rain = st.number_input("Enter the amount of rainfall recorded for the day in mm")
    st.number_input("Enter the Class A pan evaporation (mm) in the 24 hours to 9am")
    st.number_input("Enter the number of hours of bright sunshine in the day")

    wdir = st.selectbox("Enter the direction of the strongest wind gust in the 24 hours to midnight",['E', 'ENE', 'ESE', 'N', 'NE', 'NNE', 'NNW', 'NW', 'S', 'SE', 'SSE', 'SSW', 'SW', 'W','WNW', 'WSW'])
    le.classes_ = numpy.load('2.npy',allow_pickle=True)
    wdir = le.transform([wdir])

    wspeed = st.number_input("Enter the speed (km/h) of the strongest wind gust in the 24 hours to midnight")

    wdir9 = st.selectbox("Enter the Direction of the wind at 9am",['E', 'ENE', 'ESE', 'N', 'NE', 'NNE', 'NNW', 'NW', 'S', 'SE', 'SSE', 'SSW', 'SW', 'W','WNW', 'WSW'])
    le.classes_ = numpy.load('3.npy',allow_pickle=True)
    wdir9 = le.transform([wdir9])

    wdir3 = st.selectbox("Enter the Direction of the wind at 3pm",['E', 'ENE', 'ESE', 'N', 'NE', 'NNE', 'NNW', 'NW', 'S', 'SE', 'SSE', 'SSW', 'SW', 'W','WNW', 'WSW'])
    le.classes_ = numpy.load('4.npy',allow_pickle=True)
    wdir3 = le.transform([wdir3])

    wspeed9 = st.number_input("Enter the Wind speed (km/hr) averaged over 10 minutes prior to 9am.")
    wspeed3 = st.number_input("Enter the Wind speed (km/hr) averaged over 10 minutes prior to 3pm")
    humidity9 = st.number_input("Enter the Humidity (percent) at 9am")
    humidity3 = st.number_input("Enter the Humidity (percent) at 3pm")
    pres9 = st.number_input("Enter the Atmospheric pressure (hpa) reduced to mean sea level at 9am.")
    pres3 = st.number_input("Enter the Atmospheric pressure (hpa) reduced to mean sea level at 3pm")
    cloud9= st.number_input("Enter the Fraction of sky obscured by cloud (in 'oktas': eighths) at 9am")
    cloud3= st.number_input("Enter the Fraction of sky obscured by cloud (in 'oktas': eighths) at 3pm")
    temp9 = st.number_input("Enter the Temperature (degrees C) at 9am")
    temp3 = st.number_input("Enter the Temperature (degrees C) at 3pm")
    today = st.selectbox("Select yes if precipitation (mm) in the 24 hours to 9am exceeds 1mm else no",['No','Yes'])
    le.classes_ = numpy.load('5.npy',allow_pickle=True)
    today = le.transform([today])
    
    inputs = [[loc[0], mint, maxt, rain, wdir[0], wspeed, wdir9[0], wdir3[0], wspeed3, wspeed9, humidity3, humidity9, pres3, pres9, cloud3, cloud9, temp3, temp9,today]]
    inputs  = sc.transform(inputs)
    st.write(" ")

    if st.button('Predict'):
        print1(rf.predict(inputs))


if __name__=='__main__':
    main()
