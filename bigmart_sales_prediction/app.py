import streamlit as st
import pickle
from PIL import Image
import time 
import numpy 

image = Image.open('img.jpg')

lr = pickle.load(open('lr.pkl','rb'))
dt = pickle.load(open('dt.pkl','rb'))
et = pickle.load(open('et.pkl','rb'))
la = pickle.load(open('la.pkl','rb'))
rf = pickle.load(open('rf.pkl','rb'))
ri = pickle.load(open('ri.pkl','rb'))

def print1(num):
    num=num[0]
    if int(num)<0:
         num=0
    return 'Average sales prediction is ' + str(num)

def main():
    st.title("Big Mart Sales Predictor")
    st.image(image)
    st.markdown(
    """
    <style>
    .reportview-container {
        background: #f6dfeb
    }
    </style>
    """, unsafe_allow_html=True)
    activities=['LinearRegression', 'Ridge', 'Lasso','Decision Tree', 'Extra Tree Regressor', 'Random Forest']
    option=st.sidebar.selectbox('Which model would you like to use?',activities)
    st.subheader(option)
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    
    ntype = st.text_area("Enter Item Identifier")
    ntype = ntype[0:2] 
    if (ntype =='NC'):
        ntype ='Non-Consumable'
    if (ntype =='DR'):
        ntype ='Drinks'
    else:
        ntype ='Food'
    le.classes_ = numpy.load('New_Item_Type.npy',allow_pickle=True)
    ntype = le.transform([ntype])

    weight = st.number_input("Enter Item Weight",min_value =1.00)

    fat = st.selectbox("Enter Item Fat Content",('Low Fat', 'Non-Edible', 'Regular'))
    le.classes_ = numpy.load('Item_Fat_Content.npy',allow_pickle=True)
    fat = le.transform([fat])

    visible = st.number_input("Enter Item Visibility",min_value =0.00)
    
    itype = st.selectbox("Enter Item Type",('Baking Goods', 'Breads', 'Breakfast', 'Canned', 'Dairy','Frozen Foods','Fruits and Vegetables,' 'Hard Drinks', 'Health and Hygiene', 'Household', 'Meat', 'Others', 'Seafood', 'Snack Foods', 'Soft Drinks', 'Starchy Foods'))
    le.classes_ = numpy.load('Item_Type.npy',allow_pickle=True)
    itype = le.transform([itype])

    mrp = st.number_input("Enter Item's Price",min_value =0.00)
    outlet = st.selectbox("Enter Outlet Identifier",('OUT010', 'OUT013', 'OUT017', 'OUT018' ,'OUT019', 'OUT027' ,'OUT035' ,'OUT045','OUT046', 'OUT049'))
    le.classes_ = numpy.load('Outlet.npy',allow_pickle=True)
    outlet = le.transform([outlet])

    years = st.number_input("Enter establishment year",min_value =1500)
    years = 2021 - int(years)

    size = st.selectbox("Enter Outlet Size",('High', 'Medium', 'Small'))
    le.classes_ = numpy.load('Outlet_Size.npy',allow_pickle=True)
    size = le.transform([size])

    loc = st.selectbox("Enter Outlet Location",('Tier 1', 'Tier 2', 'Tier 3'))
    le.classes_ = numpy.load('Outlet_Location_Type.npy',allow_pickle=True)
    loc = le.transform([loc])

    otype = st.selectbox("Enter Outlet Type",('Grocery Store', 'Supermarket Type1', 'Supermarket Type2',
    'Supermarket Type3'))
    le.classes_ = numpy.load('Outlet_Type.npy',allow_pickle=True)
    otype = le.transform([ otype])


    inputs = [[weight, fat[0], visible, itype[0],mrp, size[0],loc[0],otype[0],ntype[0], years, outlet[0]]]

    if st.button('Predict'):
        if option=='Decision Tree':
            st.success(print1(dt.predict(inputs)))
        elif option=='LinearRegression':
            st.success(print1(lr.predict(inputs)))
        elif option=='Ridge':
            st.success(print1(ri.predict(inputs)))
        elif option=='Lasso':
            st.success(print1(la.predict(inputs)))
        elif option=='Extra Tree Regressor':
            st.success(print1(et.predict(inputs)))
        else:
            st.success(print1(rf.predict(inputs)))


if __name__=='__main__':
    main()
