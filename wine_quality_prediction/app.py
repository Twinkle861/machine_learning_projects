from google.protobuf.symbol_database import Default
import streamlit as st
import pickle
from PIL import Image
import time 

image = Image.open('img.jpg')

dec_model = pickle.load(open('decision_model.pkl','rb'))
extree_model = pickle.load(open('extra_model.pkl','rb'))
rantree_model = pickle.load(open('random_model.pkl','rb'))
light = pickle.load(open('light_model.pkl','rb'))

def classify(num):
   return "The quality on scale of 3-9 is: "+ str(num);

def main():
    st.title("Wine Quality")
    st.image(image)
    st.markdown(
    """
    <style>
    .reportview-container {
        background: #aad8d3
    }
    </style>
    """, unsafe_allow_html=True)
    activities=['Light Gradient Boosting','Decision Tree', 'Extra Tree', 'Random Forest']
    option=st.sidebar.selectbox('Which model would you like to use?',activities)
    st.subheader(option)

    
    st.text_area("type")

    fixed_acidity = st.number_input("Enter Fixed Acidity of wine",min_value=1.000,max_value=18.000)
    volatile_acidity = st.number_input("Enter Volatile Acidity of wine", min_value = 0.000, max_value = 3.000)
    citric_acid = st.number_input("Enter Citric Acid level of wine", min_value = 0.000, max_value = 3.000)
    residual_sugar = st.number_input("Enter Residual Sugar level in wine", min_value = 0.000, max_value = 70.000)
    chlorides = st.number_input("Enter Chlorides in wine", min_value = 0.000, max_value = 2.000)
    st.number_input("Enter Free Sulphur Dioxde present in wine", min_value = 0.000, max_value = 300.000)
    total_sulfur_dioxide = st.number_input("Enter Total Sulfur Dioxide of wine", min_value = 4.000, max_value = 500.000)
    density = st.number_input("Enter Density of wine", min_value = 0.000, max_value = 3.000)
    pH = st.number_input("Enter pH level of wine", min_value = 1.000, max_value = 7.000)
    sulphates = st.number_input("Enter Sulphates in wine", min_value = 0.000, max_value = 4.000)
    alcohol = st.number_input("Enter Alcohol level", min_value = 5.000, max_value = 16.000)

    inputs = [[fixed_acidity, volatile_acidity, citric_acid, residual_sugar,chlorides,total_sulfur_dioxide, density, pH, sulphates, alcohol]]
   

    if st.button('Predict'):
        if option=='Decision Tree':
            st.success(classify(dec_model.predict(inputs)))
        elif option=='Light Gradient Boosting':
            st.success(classify(light.predict(inputs)))
        elif option=='Extra Tree':
            st.success(classify(extree_model.predict(inputs)))
        else:
           st.success(classify(rantree_model.predict(inputs)))


if __name__=='__main__':
    main()
