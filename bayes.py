import sklearn

from sklearn import datasets
import pandas as pd

dataset = pd.read_csv('/Users/gienn/PycharmProjects/pythonProject/selfData')
dataset2 = pd.read_csv('/Users/gienn/PycharmProjects/pythonProject/2016-17_teamBoxScore.csv')

recentHPerformance = pd.DataFrame()
recentAPerformance = pd.DataFrame()
# print(dataset.iloc[:, 0:51])
# print(dataset.iloc[:, 51:102])


# def splitData(data):
#     recentHPerformance = data.iloc[:, 1:51]
#     recentAPerformance = data.iloc[:, 51:]
#     return recentHPerformance, recentAPerformance
# print(splitData(dataset)[1])
# # print(recentHPerformance)
# # print(recentAPerformance)
# total = pd.concat([splitData(dataset)[0], splitData(dataset)[1]])
# print(total)
# # print(recentHPerformance)
# # print(recentAPerformance)

# wine = datasets.load_wine()
#
# # print the names of the 13 features
# # print ("Features: ", wine.feature_names)
dataset1 = dataset.loc[:, ["teamAST", "teamTO", "teamSTL", "teamBLK", "teamFG%", "teamTRB", "team3P%"]]

# print the label type of wine(class_0, class_1, class_2)
# print ("Labels: ", wine.target_names)
print(dataset.shape)
# print(dataset['teamRslt'])
# print(dataset.loc[:, ["teamAST", "teamTO", "teamSTL", "teamBLK", "teamPF"]])
# print(wine.data.shape)
# print(wine.data[0:5])
# print(wine.target)
# print(wine)

import numpy as np
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(dataset1, dataset["result"], test_size=0.3,random_state=109) # 70% training and 30% test


# Split dataset into training set and test set


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
