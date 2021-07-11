import streamlit as st
import pickle
from PIL import Image
import time 
import nltk
import re
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer


genre = ['other', 'action', 'adventure', 'comedy', 'drama', 'horror', 'romance', 'sci-fi', 'thriller']

classifier = pickle.load(open('model.pkl', 'rb'))
cv = pickle.load(open('cv.pkl','rb'))
image = Image.open('img.jpg')

def classify(pred):
    # return pred
    return 'Genre is : ' + genre[pred[0]]


def main():
    st.title("Movie Genre Predictor")
    st.text("")
    st.text("")
    st.image(image)
    st.markdown(
    """
    <style>
    .reportview-container {
        background: #f5e6ca
    }
    </style>
    """, unsafe_allow_html=True)
    st.text("")
    user_input = st.text_area("Enter the plot for the movie")
    st.text("")
    if st.button('Predict'):
        time.sleep(0.5)
        if(user_input == ''):
            st.error('Please enter some plot')
        else:
            st.balloons()
            time.sleep(1)
            user_input = re.sub('[^a-zA-Z]', ' ', user_input)
            user_input = user_input.lower()
            user_input = user_input.split()
            ps = PorterStemmer()
            all_stopwords = stopwords.words('english')
            user_input = [ps.stem(word) for word in user_input if not word in set(all_stopwords)]
            user_input = ' '.join(user_input)
            corpus = [user_input]
            X = cv.transform(corpus).toarray()
            pred = classifier.predict(X)
            st.success(classify(pred))


if __name__=='__main__':
    main()