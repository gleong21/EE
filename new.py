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
y = dataset[["points"]]
x1 = dataset[["opptAST", "opptTO", "opptSTL", "opptBLK", "opptFG%", "opptTRB", "oppt3P%", "opptEFG%", "opptOREB%", "opptOrtg", "opptDrtg", "opptRslt", "opptSTL/TO", "result"]]
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

plt.scatter(X_test["teamAST"], y_test)
scale_factor = 10
# xz = [110,101,109,98,78]
# yz = [17,32,45,38,23]
# plt.xticks([90, 100, 110, 120, 130, 140])
plt.xmin = 100
# plt.xmax = 140
# ymin, ymax = plt.ylim()

# plt.xlim(xmin * scale_factor, xmax * scale_factor)
# plt.ylim(ymin * scale_factor, ymax * scale_factor)
plt.show()

# import seaborn
# sns.set_palette('colorblind')
# sns.pairplot(data=df_pie, height=3)

# for i,e in enumerate(X_train.columns):
#   regressor.fit(X_train[e].values[:,np.newaxis], y_train.values)
#   # print(X_train[e]
#
#   axes[i].set_title("Best fit line")
#   axes[i].set_xlabel(str(e))
#   axes[i].set_ylabel('SalePrice')
#   axes[i].scatter(X_train[e].values[:,np.newaxis], y_train,color='g')
#   type(axes[i].plot(X_train[e].values[:,np.newaxis], (X_train[e].values[:,np.newaxis]),color='k'))
#   fig[i].figure
#   print("checkg")

