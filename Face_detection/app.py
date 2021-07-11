import cv2
import matplotlib.pyplot as plt
import os
import streamlit as st
from PIL import Image
import io

def file_selector(folder_path='.'):
    filenames = os.listdir(folder_path)
    selected_filename = st.selectbox('Select a file', filenames)
    return os.path.join(folder_path, selected_filename)

def predict(filename):
     ##Detecting
    # HAAR Cascade File Path
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')
    # Load the Image
    image = cv2.imread(filename)
    # convert to rgb
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # plt.imshow(img_rgb)
    # resize the image
    image = cv2.resize(image, (400, 600))
    # convert to gray scale image
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # plt.imshow(gray, cmap='gray')
    # Detect Faces
    faces = face_cascade.detectMultiScale(gray)
    # len(faces)
    # diplay the faces in the image
    # diplay the faces in the image
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
    cv2.imshow("Faces ",image)
    cv2.waitKey(0)
    st.stop()
image = Image.open('img.jpg')

if __name__ == '__main__':
    st.title("Face Detector")
    st.write(" ")
    st.image(image)
    st.markdown(
    """
    <style>
    .reportview-container {
        background: #a9f1df
    }
    </style>
    """, unsafe_allow_html=True)
    # Select a file
    # if st.checkbox('Select a file in current directory'):
    #     folder_path = '.'
    #     if st.checkbox('Change directory'):
    #         folder_path = st.text_input('Enter folder path', '.')
    #     filename = file_selector(folder_path=folder_path,type='jpg,png,jpeg')
    #     st.write('You selected `%s`' % filename)
    st.write(" ")
    filename = st.text_area("Enter complete path for the iamge ")
    if st.button('Predict'):
        if filename == '':
            st.write(" ")
            st.error("Please enter path")
        else:
            try:
                st.write(" ")
                predict(filename)
                st.success("The faces are successfully detected")
            except:
                st.write(" ")
                st.warning("Please recheck file")
       
