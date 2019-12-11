from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np


dataset = load_boston()
x=dataset.data
y = dataset.target



x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=0)

#  defining model

regressor = LinearRegression()


regressor.fit(x_train,y_train)


prediction = regressor.predict(x_test)

print(np.sqrt(mean_squared_error(y_test,prediction)))
print(r2_score(y_test,prediction))


