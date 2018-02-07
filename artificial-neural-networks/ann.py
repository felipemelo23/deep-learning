#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 7 2018

@author: Felipe Melo

The dataset used here can be found at: goo.gl/iJHJZ1
"""

### Data preprocessing ###
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('data/Churn_Modelling.csv')
X = dataset.iloc[:, 3:-1].values
y = dataset.iloc[:, -1].values

# Encoding categorical data
# Encoding the Independent Variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()
X = X[:,1:]

# Splitting the dataset into the training set and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=0)

# Feature scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

### Building ANN ###
# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout

# Initialising the ANN
classifier = Sequential()

# Adding the Input Layer and the first Hidden Layer
classifier.add(Dense(units=6, kernel_initializer='uniform', activation='relu', input_dim=11))
classifier.add(Dropout(rate=0.1))

# Adding the second Hidden Layer
classifier.add(Dense(units=6, kernel_initializer='uniform', activation='relu'))
classifier.add(Dropout(rate=0.1))

# Adding the Output Layer
classifier.add(Dense(units=1, kernel_initializer='uniform', activation='sigmoid'))

# Compiling the ANN
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Fitting the ANN to the Training set
classifier.fit(X_train,y_train, batch_size=10, epochs=100)

# Saving the model
classifier.save('model/simple-ann.h5')

# Reloading the model
from keras.models import load_model
classifier = load_model('model/simple-ann.h5')

### Evaluating the Model ###
# Predict the test set results
y_pred = classifier.predict(X_test)
y_pred = np.apply_along_axis(lambda x: 1 if x > 0.5 else 0, 1, y_pred)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)

# Plotting the Confusion Matrix
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Predicting Customers Exiting (Churn)')
tick_marks = np.arange(2)
plt.xticks(tick_marks, ('Exited','Not Exited'), rotation=45)
plt.yticks(tick_marks, ('Exited','Not Exited'))

thresh = cm.max()/2

import itertools
for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    plt.text(j, i, format(cm[i, j], 'd'),
             horizontalalignment="center",
             color="white" if cm[i, j] > thresh else "black")

plt.tight_layout()
plt.ylabel('Real')
plt.xlabel('Predicted')
plt.show()

# Evaluating the ANN
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from keras.models import Sequential
from keras.layers import Dense

def build_classifier():
    classifier = Sequential()
    classifier.add(Dense(units=6, kernel_initializer='uniform', activation='relu', input_dim=11))
    classifier.add(Dense(units=6, kernel_initializer='uniform', activation='relu'))
    classifier.add(Dense(units=1, kernel_initializer='uniform', activation='sigmoid'))
    classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    return classifier

classifier = KerasClassifier(build_fn=build_classifier, batch_size=10, epochs=100)
accuracies = cross_val_score(estimator=classifier, X=X_train, y=y_train, cv=2, n_jobs=1)

mean = accuracies.mean()
variance = accuracies.std()
