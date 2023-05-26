from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

import pickle

iris = load_iris()

X=iris.data
y=iris.target


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

lr=LinearRegression()
dtc=DecisionTreeClassifier()
svc=LinearSVC(max_iter=1000)

lr.fit(X_train,y_train)
dtc.fit(X_train,y_train)
svc.fit(X_train,y_train)

pickle.dump(lr, open('ModPredLR.pkl', 'wb'))
pickle.dump(dtc, open('ModPredDTC.pkl', 'wb'))
pickle.dump(svc, open('ModPredSVC.pkl', 'wb'))
