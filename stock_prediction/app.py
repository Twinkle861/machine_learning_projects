# download stereamlit in this virtualenv again
import streamlit as st
import pickle
from PIL import Image
import time 
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from tensorflow import keras
r = keras.models.load_model('r.h5')
image = Image.open('img.jpg')


sc = pickle.load(open('sc.pkl','rb'))

def main():
    st.title("Stock Price Predictor")
    st.write(" ")
    st.write(" ")
    st.image(image,width =600)
    st.write(" ")

    st.markdown(
    """
    <style>
    .reportview-container {
        background: #867ae9
    }
    </style>
    """, unsafe_allow_html=True)
    
    activities=['Graph','Results']
    option=st.sidebar.selectbox('What would you like to see?',activities)
    st.write(" ")
    
    st.subheader("Showing results for")
    st.write(" ")
    st.subheader(option)
    st.write(" ")
    st.text("Click on show")

    dataset_test = pd.read_csv('Google_Stock_Price_Test.csv')
    dataset_train = pd.read_csv('Google_Stock_Price_Train.csv')

    real_stock_price = dataset_test.iloc[:, 1:2].values

    # Getting the predicted stock price of 2017
    dataset_total = pd.concat((dataset_train['Open'], dataset_test['Open']), axis = 0)
    inputs = dataset_total[len(dataset_total) - len(dataset_test) - 60:].values
    inputs = inputs.reshape(-1,1)
    inputs = sc.transform(inputs)
    X_test = []
    for i in range(60, 80):
        X_test.append(inputs[i-60:i, 0])
    X_test = np.array(X_test)
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    predicted_stock_price = r.predict(X_test)
    predicted_stock_price = sc.inverse_transform(predicted_stock_price)

    
    plt.plot(predicted_stock_price, color = 'blue', label = 'Predicted Google Stock Price')
    plt.title('Google Stock Price Prediction')
    plt.xlabel('Time')
    plt.ylabel('Google Stock Price')
    plt.legend()
    plt.savefig("fig.png")

    if st.button('Show'):
        if option == 'Graph':
            st.write(" ")
            im1 = Image.open('fig.png')
            st.image(im1)
        else:
            st.table(predicted_stock_price)
if __name__=='__main__':
    main()



