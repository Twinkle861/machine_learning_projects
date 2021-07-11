#issue ->app.py not working
# download stereamlit in this virtualenv again
import streamlit as st
import pickle
from PIL import Image
import time 
import matplotlib.pyplot as plt

image = Image.open('img.jpg')

model = pickle.load(open('model.pkl','rb'))

def main():
    st.title("Website Traffic Predictor")
    st.image(image,width =600)
    st.markdown(
    """
    <style>
    .reportview-container {
        background: #867ae9
    }
    </style>
    """, unsafe_allow_html=True)
    

    number = st.number_input("Enter no .of days for which you want to see the graph",1)
    number = int(number)

    if st.button('Predict'):
        if number==0:
            st.warning("Please enter a value greater than 0")
        else:
            future = model.make_future_dataframe(periods=number)
            forecast = model.predict(future)
            pred = forecast.iloc[-number:, :]
            # if st.button('Print results'):
            # print(pred.head())
            # st.write("Date                  Predicted result")
            # for i in pred:
            #     st.write(str(pred[i]['ds']) + "       "+ pred[i]['yhat'])

            # if st.button('Show Graph'):
            f = plt.figure()
            f.set_figwidth(12)
            f.set_figheight(10)
            plt.plot(pred['ds'], pred['yhat'], color='red',label='Predicted value')
            plt.plot(pred['ds'], pred['yhat_lower'], color='green',label='Lower bound of Predicted value')
            plt.plot(pred['ds'], pred['yhat_upper'], color='orange',label='Upper bound of Predicted value')
            plt.legend()
            plt.savefig('img1.jpg')
            image1 = Image.open('img1.jpg')
            st.image(image1)
            

if __name__=='__main__':
    main()



