import streamlit as st
import pickle
from PIL import Image
import time 

x = pickle.load(open('pass_agg_model.pkl','rb'))
tfidf_vectorizer = x[0]
pac = x[1]
image = Image.open('img.jpeg')

def classify(ans):
    if ans[0] == 'REAL':
        return 'News is not fake'
    else:
        return 'News is fake'

def main():
    st.title("Fake News Predictor")
    st.image(image)
    st.markdown(
    """
    <style>
    .reportview-container {
        background: #c6ffc1
    }
    </style>
    """, unsafe_allow_html=True)
    
    user_input = st.text_area("Enter the news")
    user_input = [user_input]
    # print(user_input)
    # st.text(user_input)

    if st.button('Predict'):
        time.sleep(0.5)
        if(user_input[0] == ''):
            st.error('Please enter some news')
        else:
            st.balloons()
            time.sleep(1)
            tfidf_input=tfidf_vectorizer.transform(user_input)
            st.success(classify(pac.predict(tfidf_input)))


if __name__=='__main__':
    main()