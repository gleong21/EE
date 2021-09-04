# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
import sklearn
import numpy
from sklearn.linear_model import LinearRegression
sklearn.linear_model.LinearRegression()
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score

collist = ["elo1_pre", "score1"]
collistTwo = ["elo2_pre", "score2"]
collistThree = ["score1", "score2"]
dataset = pd.read_csv('/Users/gienn/PycharmProjects/pythonProject/selfData')
datasetTwo = pd.read_csv('/Users/gienn/PycharmProjects/pythonProject/nba_elo copyOne.csv', usecols=collistTwo)
datasetThree = pd.read_csv('/Users/gienn/PycharmProjects/pythonProject/nba_elo_latest.csv', usecols=collistThree)
X4 = datasetThree.iloc[:, :-1].values
y4 = datasetThree.iloc[:, 1].values

x = dataset[["teamAST", "teamTO", "teamSTL", "teamBLK", "teamFG%", "teamTRB", "team3P%", "result"]]
y = dataset[["points"]]
x1 = dataset[["opptAST", "opptTO", "opptSTL", "opptBLK", "opptFG%", "opptTRB", "oppt3P%", "result"]]
y1 = dataset[["awayPoints"]]

# print(x)
# print(dataset.shape)
# print(datasetTwo.shape)
# dataset.plot(x, y, style='o')
# plt.title('PIE vs Win %')
# plt.xlabel('PIE')
# plt.ylabel('Win %')
# plt.show()
# datasetTwo.plot(x='elo2_pre', y="score2", style='o')
# plt.title('PIE vs Win %')
# plt.xlabel('PIE')
# plt.ylabel('Win %')
# plt.show()


# X = dataset.iloc[:, :-1].values
# y = dataset.iloc[:, 1].values
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
X1_train, X1_test, y1_train, y1_test = train_test_split(x1, y1, test_size=0.2, random_state=0)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressorOne = LinearRegression()
print(type(X_test))
regressor.fit(X_train.iloc[:, 0:6], y_train)
regressorOne.fit(X1_train.iloc[:, 0:6], y1_train)
print(regressor.intercept_)
print(regressor.coef_)

y_pred = regressor.predict(X_test.iloc[:, 0:6])
y1_pred = regressor.predict(X1_test.iloc[:, 0:6])
print('Predicted response:', y_pred, sep='\n')
print('Predicted response:', y1_pred, sep='\n')
print(X_test)
# print(y_pred.size)
# print(y1_pred.size)
winListTwo = []
for x in range(0, len(y_pred)):
    if y_pred[x] > y1_pred[x]:
        winListTwo.append("Win")
    else:
        winListTwo.append("Loss")
print(winListTwo)
print(type(X_test["result"]))
winListThree = X_test["result"]
listing = winListThree.tolist()
# winList = X_test["result"]
# print(len(winList))
# print(len(winListTwo))
# print(type(winList))
# # print(winList[1])
print(listing)
totalNum = 0
for z in range(0, len(X_test)):
    if listing[z] == winListTwo[z]:
        totalNum = totalNum + 1
print(totalNum)



# datasetOne = pd.read_csv('/Users/gienn/PycharmProjects/pythonProject/nba_elo_latest.csv', usecols=collist)
# X1 = datasetOne.iloc[:, :-1].values
# y1 = datasetOne.iloc[:, 1].values
#
# y_pred = regressor.predict(X1)
# df = pd.DataFrame({'Actual': y1, 'Predicted': y_pred})
# test = df.iloc[: , 1].values
# print(test)
# print(df)
#
# X2 = datasetTwo.iloc[:, :-1].values
# y2 = datasetTwo.iloc[:, 1].values
# regressorOne = LinearRegression()
#
# regressorOne.fit(X2, y2)
# print(regressorOne.intercept_)
# print(regressorOne.coef_)
#
# datasetThree = pd.read_csv('/Users/gienn/PycharmProjects/pythonProject/nba_elo_latest.csv', usecols=collistTwo)
# X3 = datasetThree.iloc[:, :-1].values
# y3 = datasetThree.iloc[:, 1].values
# print(X3)
# print(y3)
#
# y_predOne = regressorOne.predict(X3)
# dfOne = pd.DataFrame({'Actual': y3, 'Predicted': y_predOne})
# testOne = dfOne.iloc[: , 1].values
# print(testOne)
# # print(dfOne)
#
# winListTwo = []
# for x in range(0, len(X4)):
#     if X4[x] > y4[x]:
#         winListTwo.append("W")
#     else:
#         winListTwo.append("L")
# print(winListTwo)
#
# winList = []
# for x in range(0, len(test)):
#     if test[x] > testOne[x]:
#         winList.append("W")
#     else:
#         winList.append("L")
# print(winList)
# prediction = 0
# for x in range(0, len(winList)):
#     if winList[x] == winListTwo[x]:
#         prediction = prediction + 1
# print(prediction)
#
# print(test[32])
# print(testOne[32])









# # Load the diabetes dataset
# diabetes_X, diabetes_y = np.loadtxt('Untitled spreadsheet - Sheet1.csv', delimiter=',')
# # Use only one feature
# diabetes_X = diabetes_X[:, np.newaxis, 2]
#
# # Split the data into training/testing sets
# diabetes_X_train = diabetes_X[:-20]
# diabetes_X_test = diabetes_X[-20:]
#
# # Split the targets into training/testing sets
# diabetes_y_train = diabetes_y[:-20]
# diabetes_y_test = diabetes_y[-20:]
#
# # Create linear regression object
# regr = linear_model.LinearRegression()
#
# # Train the model using the training sets
# regr.fit(diabetes_X_train, diabetes_y_train)
#
# # Make predictions using the testing set
# diabetes_y_pred = regr.predict(diabetes_X_test)
#
# # The coefficients
# print('Coefficients: \n', regr.coef_)
# # The mean squared error
# print('Mean squared error: %.2f'
#       % mean_squared_error(diabetes_y_test, diabetes_y_pred))
# # The coefficient of determination: 1 is perfect prediction
# print('Coefficient of determination: %.2f'
#       % r2_score(diabetes_y_test, diabetes_y_pred))
#
# # Plot outputs
# plt.scatter(diabetes_X_test, diabetes_y_test,  color='black')
# plt.plot(diabetes_X_test, diabetes_y_pred, color='blue', linewidth=3)
#
# plt.xticks(())
# plt.yticks(())
#
# plt.show()
#
# # See PyCharm help at https://www.jetbrains.com/help/pycharm/
