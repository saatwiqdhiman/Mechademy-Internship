
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.metrics import r2_score



df=pd.read_csv("ailerons_train.csv")


df.head()



X=df.loc[:,df.columns.drop("goal")].values
Y=df.loc[:,"goal"].values



#split dataset into train and test

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)

scaler = StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)



model = LinearRegression().fit(X_train, y_train)



#Checking the model's accuracy
model.score(X_train, y_train)



#Predicting test set results ( predicting "goal")
y_pred = model.predict(X_test)

rmse = sqrt(mean_squared_error(y_test, y_pred))
print(" rmse value of Linear Regression is : ",rmse)
r2 = r2_score(y_test, y_pred)
print(r2)

