#! /usr/bin/env python3
# -*- coding:utf-8 -*


from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
import mglearn

X,y = mglearn.datasets.make_wave(n_samples = 40)

X_train, X_test, y_train, y_test = train_test_split(X,y,random_state=0)

reg = KNeighborsRegressor(n_neighbors=3)
reg.fit(X_train,y_train)
prediction = reg.predict(X_test)

print ("Predicition : {}".format(prediction))
print ("Score : {:.2f}".format(reg.score(X_test,y_test)))
