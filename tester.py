# given a team and a date, this method will return that teams average stats over the previous n games

import pandas as pd
dataframe = pd.read_csv('/Users/gienn/PycharmProjects/pythonProject/2012-18_teamBoxScore.csv')

dataframe["teamRslt"] = (dataframe["teamRslt"] == "Win").astype(int)
dataframe["opptRslt"] = (dataframe["opptRslt"] == "Win").astype(int)


def get_avg_stats_last_n_games(team, game_date, season_team_stats, n):
    prev_game_df = dataframe[season_team_stats['gmDate'] < game_date][
        (season_team_stats['teamAbbr'] == team)].sort_values(by='gmDate').tail(n)



    h_df = prev_game_df.iloc[:, 13:65]
    h_df.columns = [x[:] for x in h_df.columns]
    a_df = prev_game_df.iloc[:, 68:121]
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
