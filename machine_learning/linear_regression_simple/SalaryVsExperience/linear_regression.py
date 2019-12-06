import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
import matplotlib.pyplot as plt


data_dir = "Salary_Data.csv"

data = pd.read_csv(data_dir)    # reading csv file through pandas funtion

# Preprocessing

X = data.iloc [:,:-1].values
Y = data.iloc [:,1].values

# splitting data to rightly fit to the training set

x_train,x_test ,y_train, y_test = train_test_split(X,Y,test_size=.25,random_state=0) #splitting data to rightly fit to the training set

# fit data into the model(linearRegression)

lineareg = LinearRegression()
lineareg.fit(x_train , y_train)

# predicting on the basis of Test set

result= lineareg.predict(x_test)


# evaluating the quality of a model’s predictions

print(np.sqrt(metrics.mean_squared_error(y_test,result )))

# Visualising the Test set results

plt.scatter(x_test,y_test)
plt.plot(x_train,lineareg.predict(x_train))
plt.title('Salary vs Experience (Test set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()


# Visualising the train set results

plt.scatter(x_train,y_train)
plt.plot(x_train,lineareg.predict(x_train))
plt.title('Salary vs Experience (train set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()
