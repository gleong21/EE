# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
import matplotlib
import sklearn
import numpy
from sklearn.linear_model import LinearRegression
sklearn.linear_model.LinearRegression()
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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

x = dataset[["teamAST", "teamTO", "teamSTL", "teamBLK", "teamFG%", "teamTRB", "team3P%", "teamEFG%", "teamOREB%", "teamOrtg", "teamDrtg", "teamRslt", "teamSTL/TO", "result"]]
xColumnNames = ["teamAST", "teamTO", "teamSTL", "teamBLK", "teamFG%", "teamTRB", "team3P%", "teamEFG%", "teamOREB%", "teamOrtg", "teamDrtg", "teamRslt", "teamSTL/TO", "result"]

y = dataset[["points"]]
x1 = dataset[["opptAST", "opptTO", "opptSTL", "opptBLK", "opptFG%", "opptTRB", "oppt3P%", "opptEFG%", "opptOREB%", "opptOrtg", "opptDrtg", "opptRslt", "opptSTL/TO", "result"]]
xColumnNamesAway = ["opptAST", "opptTO", "opptSTL", "opptBLK", "opptFG%", "opptTRB", "oppt3P%", "opptEFG%", "opptOREB%", "opptOrtg", "opptDrtg", "opptRslt", "opptSTL/TO", "result"]

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
regressor.fit(X_train.iloc[:, 0:12], y_train)
regressorOne.fit(X1_train.iloc[:, 0:12], y1_train)
print(regressor.intercept_)
print(regressor.coef_)

y_pred = regressor.predict(X_test.iloc[:, 0:12])
y1_pred = regressor.predict(X1_test.iloc[:, 0:12])
print('Predicted response:', y_pred, sep='\n')
print('Predicted response:', y1_pred, sep='\n')
print(X_test)
winListTwo = []
for x in range(0, len(y_pred)):
    if y_pred[x] > y1_pred[x]:
        winListTwo.append(1)
    else:
        winListTwo.append(0)
print(type(X_test["result"]))
winListThree = X_test["result"]
listing = winListThree.tolist()
totalNum = 0
for z in range(0, len(X_test)):
    if listing[z] == winListTwo[z]:
        totalNum = totalNum + 1
print(totalNum)

print(regressor.score(X_test.iloc[:, 0:12], y_test))

print(X_train["teamTO"].to_numpy())
print(y_test.size)


# ax.plot(X_test["teamTO"].to_numpy(),y_pred,c='r')
# ax.set_ylim(50)
# ax.set_ylim(ymin=14)
# ax.set_xlim(xmin=70)
# ax.yticks([90, 100, 110, 120, 130, 140])

# plt.scatter(, color='black')
coeff = [1.709e-1, 5.512e-1, 6.741e-1,2.257e-2, -7.624e1, 7.586e-1, -2.595e1,4.698e1,-5.861e1,7.644e-1, 1.366e-1,-1.043]
coeffTest = []
intercept =[]
coeffTestAway = []
interceptAway = []
print(1.709e-1)

for z in range(14):
    regr = linear_model.LinearRegression()
    # print(z)
    # print(xColumnNames[z])
    x_train_changed = X_train[xColumnNames[z]]
    x_train_changed = x_train_changed.values.reshape(-1,1)
    regr.fit(x_train_changed, y_train)
    # print(regr.coef_)
    print(r2_score(x_train_changed, y_train))
    coeffTest.append(regr.coef_)
    intercept.append(regr.intercept_)
for o in range(14):
    regrOne = linear_model.LinearRegression()
    # print(z)
    # print(xColumnNamesAway[o])
    x_train_changed1 = X1_train[xColumnNamesAway[o]]
    x_train_changed1 = x_train_changed1.values.reshape(-1,1)
    regrOne.fit(x_train_changed, y1_train)
    # print(regrOne.coef_)
    coeffTestAway.append(regrOne.coef_)
    interceptAway.append(regrOne.intercept_)
# ymin, ymax = plt.ylim()
print(coeffTest[1][0])
ax = plt.subplot(2, 1, 1)
# m, b = np.linearfit(X_test["teamTO"].to_numpy(),y_test.to_numpy(), 1)
plt.scatter(X_train["teamDrtg"].to_numpy(),y_train.to_numpy(),c='r')
print(type(X_train["teamDrtg"].to_numpy()))
testingArray = np.array(X_train["teamDrtg"])
plt.plot(testingArray, (coeffTest[10][0]*testingArray + intercept[10][0]))
plt.title("teamDrtg vs Points")

ax.set_xlabel("teamDrtg")
ax.set_ylabel("Points")

fig = matplotlib.pyplot.gcf()

fig.set_size_inches(12.5, 10.5, forward=True)

# plt.show()

ax1 = plt.subplot(2, 1, 2)
# m, b = np.linearfit(X_test["teamTO"].to_numpy(),y_test.to_numpy(), 1)
plt.scatter(X_train["teamRslt"].to_numpy(),y_train.to_numpy(),c='r')
print(type(X_train["teamRslt"].to_numpy()))
testingArray = np.array(X_train["teamRslt"])
ax1.set_xlabel("teamRslt")
ax1.set_ylabel("Points")
plt.plot(testingArray, (coeffTest[11][0]*testingArray + intercept[11][0]))
plt.title("teamRslt vs Points")
fig = matplotlib.pyplot.gcf()

fig.tight_layout()
plt.savefig("plot6.png")

plt.show()




