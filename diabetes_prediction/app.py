
import pickle
from PIL import Image
import time 
import numpy as np
import streamlit as st

model = pickle.load(open('model.pkl','rb'))
image = Image.open('img.jpg')

def main():
    st.title("Diabetes Predictor")
    st.image(image, width=500)
    st.markdown(
    """
    <style>
    .reportview-container {
        background: #c7ffd8
    }
    image{
        align : center
    }
    </style>
    """, unsafe_allow_html=True)
    
    bmi = st.number_input('Enter your BMI',min_value = 13.00)
    preg = st.number_input('Enter no. of Pregnancies',min_value=0)
    glucose = st.number_input('Enter Glucose Level',min_value = 30)
    bp = st.number_input('Enter your Blood Pressure',min_value = 5)
    sth = st.number_input('Enter your Skin Thickness',min_value = 5)
    insulin = st.number_input('Enter your Insulin Level',min_value = 10)
    age = st.number_input('Enter your Age',min_value=1)
    dpf= st.number_input('Enter your Diabetes Pedigree Function')

    if st.button('Predict'):
        time.sleep(0.5)
        st.balloons()
        time.sleep(1)
        data = np.array([[preg, glucose, bp, sth, insulin, bmi, dpf, age]])
        ans = model.predict(data)
        if ans[0] == 1:
            st.warning('You can have diabetes')
        else:
            st.success('You donot have diabetes')



if __name__=='__main__':
    main()