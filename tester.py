# given a team and a date, this method will return that teams average stats over the previous n games

import pandas as pd
dataframe = pd.read_csv('/Users/gienn/PycharmProjects/pythonProject/2012-18_teamBoxScore.csv')

def get_avg_stats_last_n_games(team, game_date, season_team_stats, n):
    prev_game_df = dataframe[season_team_stats['gmDate'] < game_date][
        (season_team_stats['teamAbbr'] == team)].sort_values(by='gmDate').tail(n)



    h_df = prev_game_df.iloc[:, 15:65]
    h_df.columns = [x[:] for x in h_df.columns]
    a_df = prev_game_df.iloc[:, 70:121]
    # print(a_df)
    # print(h_df)
    a_df.columns = [x[:] for x in a_df.columns]

    df = pd.concat([h_df, a_df])
    # df = df[df['teamAbbr'] == team]
    # df.drop(columns=['teamAbbr'], inplace=True)

    return df.mean()


recent_performance_df = pd.DataFrame()

# print(get_avg_stats_last_n_games("ATL", "2017-02-01" , dataframe, 10))
dataframeTester = pd.DataFrame()
for index, row in dataframe.iterrows():
    if(index > 500):
        k = get_avg_stats_last_n_games(row["teamAbbr"], row["gmDate"] , dataframe, 10)
        k = k.to_frame().transpose()
        k["result"] = row['teamRslt']
        k["Home"] = row["teamAbbr"]
        k["Away"] = row["opptAbbr"]
        k["points"] = row["teamPTS"]
        k["awayPoints"] = row["opptPTS"]
        k["date"] = row["gmDate"]
        dataframeTester = dataframeTester.append(k)

dataframeTester.to_csv("/Users/gienn/PycharmProjects/pythonProject/selfData")
from sklearn import datasets
import pandas as pd

dataset = pd.read_csv('/Users/gienn/PycharmProjects/pythonProject/2012-18_teamBoxScore.csv')
dataset2 = pd.read_csv('/Users/gienn/PycharmProjects/pythonProject/2016-17_teamBoxScore.csv')

wine = datasets.load_wine()

# print the names of the 13 features
# print ("Features: ", wine.feature_names)
dataset1 = dataframeTester.loc[:, ["teamAST", "teamTO", "teamSTL", "amBLK", "amFG%", "amTRB", "am3P%"]]

# print the label type of wine(class_0, class_1, class_2)
# print ("Labels: ", wine.target_names)
print(dataframeTester.shape)
# print(dataset['teamRslt'])
# print(dataset.loc[:, ["teamAST", "teamTO", "teamSTL", "teamBLK", "teamPF"]])
# print(wine.data.shape)
# print(wine.data[0:5])
# print(wine.target)
# print(wine)

import numpy as np
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(dataset1, dataframeTester["result"], test_size=0.3,random_state=109) # 70% training and 30% test


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
print(y_pred)

# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))