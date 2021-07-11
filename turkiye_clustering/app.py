
import streamlit as st
import pickle
from PIL import Image

c1 = pickle.load(open('c1.pkl','rb'))
c2 = pickle.load(open('c2.pkl','rb'))
# print(c1)
# print(type(c1))
image = Image.open('img.jpg')
k1 = Image.open('k1.jpg')
k2 = Image.open('k2.jpg')
d1 = Image.open('d1.jpg')
d2 = Image.open('d2.jpg')

def main():
    st.title("Turkiye Student Evaluation")
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
    activities=['K Means','Hierarchial Clustering']
    option=st.selectbox('Which model would you like to use?',activities)

    if st.button('Show Results'):
        if option=='K Means':
            st.write("Following is the result of elbow method")
            st.write(" ")
            st.image(k1)
            st.write(" ")
            st.write("Hence, we select 3 clusters")
            st.write("We get following graph")
            st.write(" ")
            st.image(k2)
            st.write(" ")
            st.write("We can draw following inferences")
            op = "The dataset is divides into 3 clusters and no. of students in "
            for k,v in c1.items():
                op = op + " cluster" + str(k) +" : " + str(v) +"    " 
            st.success(op)
        else:
            st.write("Following is the result of dendrogram method")
            st.write(" ")
            st.image(d1)
            st.write(" ")
            st.write("Hence, we select 2 clusters")
            st.write("We get following graph")
            st.write(" ")
            st.image(d2)
            st.write(" ")
            st.write("We can draw following inferences")
            op = "The dataset is divides into 2 clusters and no. of students in\n"
            for k,v in c2.items():
                op = op + " in cluster" + str(k) +" : " + str(v) +"\n" 
            st.success(op)
           


if __name__=='__main__':
    main()



