import streamlit as st
import pickle
from PIL import Image
import numpy 

image = Image.open('img.jpg')

lr = pickle.load(open('lr.pkl','rb'))
dt = pickle.load(open('dt.pkl','rb'))
et = pickle.load(open('et.pkl','rb'))
rf = pickle.load(open('rf.pkl','rb'))

def print1(num):
    num=num[0]
    if int(num)<0:
         num=0
    return 'Average purchase prediction is ' + str(num)
print("X")
def main():
    st.title("Black Friday Sales Predictor")
    st.image(image,width = 500)
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
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    
    st.text_input("Enter User ID")
    st.text_input("Enter Product ID")
    
    gender = st.selectbox("Enter Gender",('F', 'M'))
    le.classes_ = numpy.load('Gender.npy',allow_pickle=True)
    gender = le.transform([gender])

    age = st.selectbox("Enter Age",('0-17', '18-25', '26-35', '36-45', '46-50', '51-55', '55+'))
    le.classes_ = numpy.load('age.npy',allow_pickle=True)
    age = le.transform([age])

    occ = st.number_input("Enter number of occupations",min_value =0)

    city = st.selectbox("Enter City Category",('A', 'B', 'C'))
    le.classes_ = numpy.load('City_Category.npy',allow_pickle=True)
    city = le.transform([city])


    stay = st.selectbox("Enter number of years customer has been staying in this city",('0', '1', '2', '3', '4+'))
    le.classes_ = numpy.load('Stay_In_Current_City_Years.npy',allow_pickle=True)
    stay = le.transform([stay])


    mar = st.number_input("Enter marital status",min_value=0)
    pr1 = st.number_input("Enter product category 1",min_value=0.0)
    pr2 = st.number_input("Enter product category 2",min_value=0.0)
    pr3 = st.number_input("Enter product category 3",min_value=0.0)

 

    inputs = [[gender, age[0], occ, city[0], stay[0], mar, pr1, pr2, pr3 ]]

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
