# #traffic forecast on website->
# fbprophet=time series analysis
# resample('D)->combine all data for 1 day
# .tail() displays last 5 rows
# future prediction contains training data too therfore select last rows
#to download fbprophet need to craete a virtual env->conda create -n myenv python=3.7
# activate it->source actiavte myenv
# in cmd to activate we type =>conda activate myenv
# then install fbprophet->conda install -c anaconda ephem
# conda install -c conda-forge pystan
# conda install -c conda-forge fbprophet
#to run python in git bash=>python -i


import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')
from fbprophet import Prophet

df = pd.read_csv('Traffic data.csv')
df['Datetime'] = pd.to_datetime(df['Datetime'], format='%d-%m-%Y %H:%M')
df.index = df['Datetime']
df['y'] = df['Count']
df.drop(columns=['ID', 'Datetime', 'Count'], axis=1, inplace=True)
df = df.resample('D').sum()

df['ds'] = df.index
model = Prophet(yearly_seasonality=True, seasonality_prior_scale=0.9)
model.fit(df)

pickle.dump(model,open('model.pkl','wb'))

future = model.make_future_dataframe(periods=60)
forecast = model.predict(future)
forecast.head()
pred = forecast.iloc[-60:, :]

# test results
plt.figure(figsize=(10,7))
plt.plot(pred['ds'], pred['yhat'], color='red')
plt.plot(pred['ds'], pred['yhat_lower'], color='green')
plt.plot(pred['ds'], pred['yhat_upper'], color='orange')
plt.show()
plt.plot(pred['ds'], pred['yhat'])
plt.show()