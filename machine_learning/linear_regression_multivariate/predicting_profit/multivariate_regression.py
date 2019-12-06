from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = 'STARTUP_DATA.csv'

data = pd.read_csv(data)

x = data.iloc[:,:-1]
y = data.iloc[:,-1]

encoded = pd.get_dummies(x['State'])
x = x.drop('State', axis = 1)

x = pd.concat([x,encoded], axis = 1)
print(x)
x_train, x_test, y_train,  y_test = train_test_split(x,y,test_size=0.20,random_state=0)

mutlivariate_Regression = LinearRegression()

mutlivariate_Regression.fit(x_train, y_train)

prediction = mutlivariate_Regression.predict(x_test)

result = r2_score(y_true=y_test,y_pred=prediction)

print(result)

