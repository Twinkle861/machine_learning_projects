# Apriori

# Importing the libraries
import numpy as np
from PIL import Image
import pandas as pd
import streamlit as st

 # Data Preprocessing
dataset = pd.read_csv('Market_Basket_Optimisation.csv', header=None)
transactions = []
for i in range(0, 7501):
  transactions.append([str(dataset.values[i, j]) for j in range(0, 20)])

# Training the Apriori model on the dataset
from apyori import apriori
rules = apriori(transactions=transactions, min_support= 0.003, min_confidence = 0.2, min_lift = 3,min_length = 2, max_length = 2)

# Visualising the results

# Displaying the first results coming directly from the output of the apriori function
results = list(rules)

# Putting the results well organised into a Pandas DataFrame
def inspect(results):
    lhs = [tuple(result[2][0][0])[0] for result in results]
    rhs = [tuple(result[2][0][1])[0] for result in results]
    supports = [result[1] for result in results]
    confidences = [result[2][0][2] for result in results]
    lifts = [result[2][0][3] for result in results]
    return list(zip(lhs, rhs, supports, confidences, lifts))
resultsinDataFrame = pd.DataFrame(inspect(results), columns=[
                                      'Left Hand Side', 'Right Hand Side', 'Support', 'Confidence', 'Lift'])

# ## Displaying the results non sorted
# resultsinDataFrame

# ## Displaying the results sorted by descending lifts
# resultsinDataFrame.nlargest(n = 10, columns = 'Lift')
image = Image.open('img.png')


def main():
    st.title("Best Deal Predictor")
    st.image(image, width=600)
    st.markdown(
        """
    <style>
    .reportview-container {
        background: #F6C6EA
    }
    </style>
    """, unsafe_allow_html=True)

    number = st.number_input("Enter no .of top deals you want to see", 1)
    number = int(number)

    if st.button('Predict'):
        if number == 0:
            st.warning("Please enter a value greater than 0")
        else:
          x = resultsinDataFrame.nlargest(n = number, columns = 'Lift').to_numpy()
          if number < len(x):
            number = number 
          else:
             number = len(x)  
          for i in range(0,number):
            y = str(i+1) + " : Keep " + str(x[i][0]) + " with " + str(x[i][1]) + ", it has lift of " + str(x[i][4]) + "."
            st.success(y)

if __name__ == '__main__':
    main()
