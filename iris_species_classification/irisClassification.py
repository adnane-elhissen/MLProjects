#! /usr/sys/env python3
# -*- coding:utf-8 -*

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import pandas as pd
import mglearn
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
import numpy as np


iris_data = load_iris()
keys = iris_data.keys() 
print ("keys of iris_dataset : {}".format(keys))
print ("Possible classes of iris : {}".format(iris_data['target_names']))


X_train,X_test,y_train,y_test = train_test_split (iris_data['data'],iris_data['target'],random_state =0)
iris_dataframe = pd.DataFrame (X_train,columns = iris_data['feature_names'])
#print (iris_dataframe)
#pd.plotting.scatter_matrix(iris_dataframe,c=y_train,figsize=(15,15),marker='o',hist_kwds={'bins':20},s=60,alpha=.8,cmap=mglearn.cm3)
#plt.show()
knn = KNeighborsClassifier (n_neighbors=1)
knn.fit(X_train,y_train)

X_new = np.array([[5,2.9,1,0.2]])
print (X_new.shape)
prediction = knn.predict(X_new)

print ("Predicted class is : {}".format(iris_data['target_names'][prediction]))
print ("Test score : {}".format(knn.score(X_test,y_test)))

