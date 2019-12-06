from sklearn.model_selection import train_test_split
from sklearn import neighbors
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
from sklearn.datasets import load_iris

# Load and return the iris dataset in variable(dataset)
dataset = load_iris()

x = dataset.data
y = dataset.target

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=0)

knn_classifier = neighbors.KNeighborsClassifier(n_neighbors=2)

# fit data into the model
knn_classifier.fit(x_train , y_train)


# just a random prediction on a set of features
# import numpy as np
# random_test = np.array([[5.1,3.5,1.4,0.2]])
# print(knn_classifier.predict(random_test))

# predicting on the basis of Test set
prediction = knn_classifier.predict(x_test)

# evaluating the quality of a model’s predictions
print(accuracy_score(y_true= y_test, y_pred=prediction))
print(confusion_matrix(y_test,prediction))
print(classification_report(y_test,prediction))

