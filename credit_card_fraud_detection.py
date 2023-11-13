# -*- coding: utf-8 -*-
"""
Created on Sun Nov 12 16:41:43 2023

@author: saikr
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report 
from sklearn.metrics import accuracy_score
# loading the dataset to a Pandas DataFrame
df = pd.read_csv('C:/Users/saikr/OneDrive/Desktop/creditcard.csv')
# plotting the data on different histograms
df.hist(bins=30,figsize =(30,30))
#describing the data
print(df.describe())
# first 5 rows of the dataset
print(df.head())
print(df.tail())
# dataset informations
print(df.info())
# checking the number of missing values in each column
print(df.isnull().sum())
# distribution of legit transactions & fraudulent transactions
print(df['Class'].value_counts())
# separating the data for analysis
legit = df[df.Class == 0]
fraud = df[df.Class == 1]
print(legit.shape)
print(fraud.shape)
# statistical measures of the data
print(legit.Amount.describe())
print(fraud.Amount.describe())
# compare the values for both transactions
print(df.groupby('Class').mean())
legit_sample = legit.sample(n=492)  
#creating a new dataset
new_dataset = pd.concat([legit_sample, fraud], axis=0)
print(new_dataset.head())
print(new_dataset.tail())
print(new_dataset['Class'].value_counts())
#grouping the dataset by ckass and calculating the mean.
print(new_dataset.groupby('Class').mean())
X = new_dataset.drop(columns='Class', axis=1)
Y = new_dataset['Class']    
print(X)
print(Y)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)
print(X.shape, X_train.shape, X_test.shape)
print(Y.shape, Y_train.shape, Y_test.shape)
model = LogisticRegression()
# training the Logistic Regression Model with Training Data
model.fit(X_train, Y_train)
# accuracy on training data
X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)
print('Accuracy on Training data : ', training_data_accuracy)
#printing a classification report on our training data
print(classification_report(Y_test, model.predict(X_test),target_names=['Not fraud','Fraud']))
# accuracy on test data, this accuracy is more important to consider than the training accuracy. 
#There should not be any major difference between the accuracies. 
#If major difference is seen , it can be either under fitting or over fitting.
X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)
print('Accuracy score on Test Data : ', test_data_accuracy)

