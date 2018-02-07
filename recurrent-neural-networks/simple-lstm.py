#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  7 09:34:26 2018

@author: Felipe Melo
"""
### Preprocessing Training Data ###
# Importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Importing the train data
dataset_train = pd.read_csv('data/Google_Stock_Price_Train.csv')
trainset = dataset_train.iloc[:,1:2].values 

# Feature Scaling
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))
trainset_scaled = scaler.fit_transform(trainset)

# Creating a data structure with 60 timesteps and 1 output
train_X,train_y = [],[]

for t in range(60, 1258):
    train_X.append(trainset_scaled[t-60:t,0])
    train_y.append(trainset_scaled[t,0])
    
train_X, train_y = np.array(train_X), np.array(train_y)

# Reshaping 
train_X = np.reshape(train_X, (train_X.shape[0],train_X.shape[1],1))

### Building the LSTM ###
# Importing the Keras libraries and packages
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

# Configuring the GPU
config = tf.ConfigProto( device_count = {'GPU': 1 , 'CPU': 2} ) 
sess = tf.Session(config=config) 
keras.backend.set_session(sess)

# Initializing the LSTM
regressor = Sequential()

# Adding the first LSTM layers and some Dropout regularization
regressor.add(LSTM(units=50,return_sequences=True, input_shape=(train_X.shape[1],1)))
regressor.add(Dropout(rate=0.2))

# Adding a second LSTM layers and some Dropout regularization
regressor.add(LSTM(units=50,return_sequences=True))
regressor.add(Dropout(rate=0.2))

# Adding a third LSTM layers and some Dropout regularization
regressor.add(LSTM(units=50,return_sequences=True))
regressor.add(Dropout(rate=0.2))

# Adding a fourth LSTM layers and some Dropout regularization
regressor.add(LSTM(units=50))
regressor.add(Dropout(rate=0.2))

# Adding the output layer
regressor.add(Dense(units=1))

# Compiling the LSTM
regressor.compile(optimizer='adam', loss='mean_squared_error')

# Fitting the LSTM to the training set
regressor.fit(train_X, train_y, epochs=100, batch_size=32)

# Saving the fitted model
regressor.save('model/simple_lstm.h5')

# Loading the the model back
from keras.models import load_model
regressor = load_model('model/simple_lstm.h5')

### Visualizing the Results ###
# Getting the real stock price of 2017
dataset_test = pd.read_csv('data/Google_Stock_Price_Test.csv')
real_stock_price = dataset_test.iloc[:,1:2].values

# Getting the predicted stock price of 2017
dataset_total = pd.concat((dataset_train['Open'], dataset_test['Open']), axis=0)
inputs = dataset_total[len(dataset_total)-len(dataset_test)-60:].values
inputs = inputs.reshape(-1,1)
inputs = scaler.transform(inputs)

test_X = []
for t in range(60, 80):
    test_X.append(inputs[t-60:t,0])
test_X = np.array(test_X)
test_X = np.reshape(test_X, (test_X.shape[0], test_X.shape[1], 1))

pred_scaled_y = regressor.predict(test_X)
pred_stock_price = scaler.inverse_transform(pred_scaled_y)

# Visualising the results
plt.plot(real_stock_price, color='red', label='Real Google Stock Price')
plt.plot(pred_stock_price, color='purple', label='Predicted Google Stock Price')
plt.title('Google Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Google Stock Price')
plt.legend()
plt.show()





