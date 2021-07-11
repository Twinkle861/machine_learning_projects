import streamlit as st
import pickle
from PIL import Image
import time 

image = Image.open('img.jpeg')

log_model = pickle.load(open('log_model.pkl','rb'))
dec_model = pickle.load(open('dec_model.pkl','rb'))
extree_model = pickle.load(open('extree_model.pkl','rb'))
rantree_model = pickle.load(open('rantree_model.pkl','rb'))
le = pickle.load(open('label_encoder.pkl','rb'))

def classify(num):
    if num == 0:
        return 'Not approved'
    else:
        return 'Aprroved'

def main():
    st.title("Loan Predictor")
    st.image(image)
    st.markdown(
    """
    <style>
    .reportview-container {
        background: #f6dfeb
    }
    </style>
    """, unsafe_allow_html=True)
    activities=['Logistic Regression','Decision Tree', 'Extra Tree', 'Random Forest']
    option=st.sidebar.selectbox('Which model would you like to use?',activities)
    st.subheader(option)

    st.text_area("Enter Loan_id")

    gender = st.selectbox("Enter your gender",("Male","Female"))
    gender = le.fit_transform([gender])

    married = st.selectbox("Are you married",("Yes","No"))
    married = le.fit_transform([married])

    dependents = st.selectbox("Enter no .of dependents",('0','1','2','3'))

    education = st.selectbox("Enter your education",("Graduate","Not Graduate"))
    education = le.fit_transform([education])

    self_employed = st.selectbox("are you self-employed",("Yes","No"))
    self_employed = le.fit_transform([self_employed])

    st.text_area("Enter apllicant income")
    st.text_area("Enter co-applicant income")
    st.text_area("Enter loan amount")
    st.text_area("Enter loan amount term")

    credit_history = st.selectbox("Enter Credit History",("0","1"))

    property_area = st.selectbox("Enter Property Area",("Urban","Semi-Urban","Rural"))
    property_area = le.fit_transform([property_area])

    inputs = [[gender[0], married[0], dependents, education[0],self_employed[0], credit_history, property_area[0]]]
    #cols = [0,1,3,4,6]
    #for col in cols:
     #   inputs[0][col] = le.fit_transform(inputs[0][col])

    if st.button('Predict'):
        if option=='Decision Tree':
            st.success(classify(dec_model.predict(inputs)))
        elif option=='Logistic Regression':
            st.success(classify(log_model.predict(inputs)))
        elif option=='Extra Tree':
            st.success(classify(extree_model.predict(inputs)))
        else:
           st.success(classify(rantree_model.predict(inputs)))


if __name__=='__main__':
    main()
