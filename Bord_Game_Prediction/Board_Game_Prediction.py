#importing librery
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.model_selection as train_test_split

#Load dataset
games=pd.read_csv("games.csv")
#Remove rows without any reviews
games=games[games['users_rated']>0]
#Removing missing value
games=games.dropna(axis=0)
#get all the columns on the list
columns=games.columns.tolist()
#drop the to data do not want
columns=[c for c in columns if c not in['id','type','name',
                                        'bayes_average_rating','average_rating']]
#store the variable predicting on
target='average_rating'
#Genrating the train and test dataset
train=games.sample(frac=0.8,random_state=1)
#select the data not in train dataset
test=games.loc[~games.index.isin(train.index)]
#importing linear regression model
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
LR=LinearRegression()
LR.fit(train[columns],train[target])

#Genertaing Prediction from the test set
predict=LR.predict(test[columns])
#compute error between ourtest predictions and actual values
mean_squared_error(predict,test[target])