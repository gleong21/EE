import sklearn

from sklearn import datasets
import pandas as pd

dataset = pd.read_csv('/Users/gienn/PycharmProjects/pythonProject/selfData')
dataset2 = pd.read_csv('/Users/gienn/PycharmProjects/pythonProject/2016-17_teamBoxScore.csv')

recentHPerformance = pd.DataFrame()
recentAPerformance = pd.DataFrame()
dataset1 = dataset.loc[:, ["teamAST", "teamTO%", "teamBLK", "teamEFG%", "team3P%", "teamFT%","teamOREB%", "teamOrtg", "teamDrtg", "teamFTM", "teamFT%", "teamRslt", "teamSTL/TO"]]

print(dataset.shape)


import numpy as np
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(dataset1, dataset["result"], test_size=0.2,random_state=0) # 70% training and 30% test

print(X_test.shape)


#Import Gaussian Naive Bayes model
from sklearn.naive_bayes import GaussianNB

#Create a Gaussian Classifier
gnb = GaussianNB()

#Train the model using the training sets
gnb.fit(X_train, y_train)

#Predict the response for test dataset
y_pred = gnb.predict(X_test)

#Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics
print(dataset["result"])
print(y_pred)

# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
