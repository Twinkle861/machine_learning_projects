import streamlit as st
import pickle
from PIL import Image


lg = pickle.load(open('lg.pkl','rb'))
tr = pickle.load(open('tr.pkl','rb'))
bn = pickle.load(open('bn.pkl','rb'))
# dt = pickle.load(open('dt.pkl','rb'))
# rf = pickle.load(open('rf.pkl','rb'))
# ml = pickle.load(open('ml.pkl','rb'))
image = Image.open('img.jpg')

def classify(num):
    num = int(num[0])
    if num == 0:
        st.error("Your password is weak.")
    elif num == 1:
        st.warning("Your password is moderate.")
    else:
        st.success("Your password is strong.")
def main():
    st.title("Password Strength Checker")
    st.write(" ")
    st.write(" ")
    st.image(image)
    st.write(" ")
    st.markdown(
    """
    <style>
    .reportview-container {
        background: #5F939A
    }
    </style>
    """, unsafe_allow_html=True)
    activities=['Logistic Regression','Naive Bayes','Decision Tree', 'Random Forest', 'MLP Classifier']
    option=st.sidebar.selectbox('Which model would you like to use?',activities)
    st.write(" ")
    st.subheader(option)
    passw = st.text_input("Enter your password")
    
    inputs = tr.transform([passw]) 
    # inputs=[[inputs]]
    if st.button('Classify'):
        if option=='Logistic Regression':
            classify(lg.predict(inputs))
        elif option=='Naive Bayes':
            classify(bn.predict(inputs))
        # elif option=='Decision Tree':
        #     classify(dt.predict(inputs))
        # elif option=='Random Forest':
        #     classify(rf.predict(inputs))
        # elif option=='MLP Classifier':
        #     classify(ml.predict(inputs))
        

if __name__=='__main__':
    main()