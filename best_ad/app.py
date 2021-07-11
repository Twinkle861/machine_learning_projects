from re import I
import streamlit as st
import pickle
from PIL import Image


image = Image.open('img.webp')

ad = pickle.load(open('ad.pkl','rb'))
d = pickle.load(open('d.pkl','rb'))

def main():
    st.title("Best Ad Predictor")
    st.write(" ")
    st.write(" ")
    st.write(" ")
    st.image(image)
    st.write(" ")
    st.write(" ")
    st.markdown(
    """
    <style>
    .reportview-container {
        background: #94d0cc
    }
    </style>
    """, unsafe_allow_html=True)
        
    activities=['Count of each ad','Graphical results(Histogram)']
    option=st.sidebar.selectbox('What would you like to see?',activities)
    
    st.subheader("Showing results for :")
    st.subheader(option)
    st.write(" ")
    st.write(" ")
    if st.button('Show Results'):
        if option=='Graphical results(Histogram)':
            image1 = Image.open('x.jpg')
            st.write(" ")
            st.write(" ")
            st.write(" ")

            st.image(image1)

        else:
            st.write(" ")
            max = 0
            a=0
            for i in range (0,d):
                if(ad.count(i)>max):
                    max = ad.count(i)
                    a=i
                st.warning("Count of ad "+str(i)+" is "+str(ad.count(i)) + ".")
            st.success("Maximum pepople liked ad " + str(a) + " ie." + str(max) + " people .")


if __name__=='__main__':
    main()
