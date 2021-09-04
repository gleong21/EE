import pandas as pd
import numpy as np
dataframe = pd.read_csv('/Users/gienn/PycharmProjects/pythonProject/2012-18_teamBoxScore.csv')

print(dataframe.iloc[:, 15:65])
print(dataframe.iloc[:, 70:121])