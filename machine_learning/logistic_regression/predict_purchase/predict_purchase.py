import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix

data = 'socialnetworkAd.csv'

data = pd.read_csv(data)

# preprocessing
x=data.iloc[:,2:4]
y = data.iloc[:,-1]

# feature scaling
sc = StandardScaler()
x = sc.fit_transform(x)

# spliting data to fit in train and test set
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.25,random_state=0)

# defining model
logit_reg = LogisticRegression(random_state=0)

logit_reg.fit(x_train,y_train)

# prediction o the basis of x_test
result = logit_reg.predict(x_test)
print(result)

#performance of a classification model
print(confusion_matrix(y_test,result))



