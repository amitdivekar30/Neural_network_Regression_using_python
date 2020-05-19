# -*- coding: utf-8 -*-
"""
Created on Tue May 19 13:19:38 2020

@author: aad
"""
import numpy as np
import pandas as pd

from keras.models import Sequential
from keras.layers import Dense, Activation,Layer,Lambda

# Reading data 
dataset = pd.read_csv("50_Startups.csv")
dataset.head()
dataset.columns
dataset.describe()
dataset.info()

# creating dummy columns for the categorical columns 
dataset.columns
dummies = pd.get_dummies(dataset[["State"]])
# Dropping the columns for which we have created dummies
dataset.drop(["State"],inplace=True,axis = 1)

# adding the columns to the dataset data frame 
dataset = pd.concat([dataset,dummies],axis=1)
dataset.columns
dataset.head(10)


def prep_model(hidden_dim):
    model = Sequential()
    for i in range(1,len(hidden_dim)-1):
        if (i==1):
            model.add(Dense(hidden_dim[i],input_dim=hidden_dim[0],kernel_initializer="normal",activation="relu"))
        else:
            model.add(Dense(hidden_dim[i],activation="relu"))
    model.add(Dense(hidden_dim[-1]))
    model.compile(loss="mean_squared_error",optimizer="adam",metrics = ["accuracy"])
    return (model)

column_names = list(dataset.columns)
target = column_names[3]
column_names.remove('Profit')
predictors = column_names

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
dataset.iloc[:, 0:3] = sc.fit_transform(dataset.iloc[:, 0:3])
print(X)

model_NN = prep_model([len(predictors),50,1])
model_NN.fit(np.array(dataset[predictors]),np.array(dataset[target]),epochs=900)
pred_train = model_NN.predict(np.array(dataset[predictors]))
pred_train = pd.Series([i[0] for i in pred_train])
rmse_value = np.sqrt(np.mean((pred_train-dataset[target])**2))
rmse_value
import matplotlib.pyplot as plt
plt.plot(pred_train,dataset[target],"bo")
np.corrcoef(pred_train,dataset[target]) # we got high correlation 


