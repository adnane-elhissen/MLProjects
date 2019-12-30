#! /usr/sys/env python 3

from KNearestNeighbors import KNearestNeighbors
import data_exploration  as de
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import mglearn

import pandas as pd
import numpy as np 

def process_classification_by_sickit_learn():
    #load iris data from sklearn
    iris_dataset = load_iris()
    
    #split dataset into two parts : training datas and test datas
    X_train,X_test,y_train,y_test = train_test_split (iris_dataset['data'],iris_dataset['target'],random_state =0)

    #display and plot to verify
    iris_dataframe = pd.DataFrame (X_train,columns = iris_dataset['feature_names'])
    pd.plotting.scatter_matrix(iris_dataframe,c=y_train,figsize=(15,15),marker='o',hist_kwds={'bins':20},s=60,alpha=.8,cmap=mglearn.cm3)
    plt.show()  
    
    #Classification with K Nearest Neighbors (kNN)
    knn = KNeighborsClassifier(n_neighbors=1)
    knn.fit(X_train,y_train)

    #predict with new value
    X_new = np.array([[5,2.9,1,0.2]])
    prediction = knn.predict(X_new)
    print ("Predicted class is : {}".format(iris_dataset['target_names'][prediction]))
    print ("Test score : {}".format(knn.score(X_test,y_test)))

    
def process_classification_by_custom_class():
    
 # loading data set CSV in data_sets repository using load_data_set function in data_exploration.py
    l = de.load_data_set("data_sets/iris_data_sets.csv")
    species = []
    vector_petal_length_width = []
    for i,ele in enumerate(l):
        if i==0 : 
            continue
        else :
            #Convert string value to float value for mathematical operations
            vector_petal_length_width.append((float(ele[0]),float(ele[1])))
            species.append(float(ele[2]))
    # Recover data with vectors and associated label for each vector
    # In this case vector = petal_length,petal_width
    # label : species = 0 "Setosa" , 1 "Verginica" , 2 "Versicolor"         
    data = list(zip (vector_petal_length_width,species))
    length = 2.5
    width = 0.75
    point =(length,width)
    KNN = KNearestNeighbors(5,data,point)
    cl = KNN.finalPrediction()
    
    #Split data by two for training and test performance : scoring
    training_pct = 0.8*len(vector_petal_length_width) # proportion of data for training 
    # We use for this example 80% for training
    vector_training,vector_test = de.split_data(data,training_pct)
    X_train,y_train = zip(*vector_training)
    X_test,y_test = zip(*vector_test)
    
    training_data_ = list(zip (list(X_train),list(y_train)))
    test_data_ = list(zip (list(X_test),list(y_test)))
    num_test_data = len (test_data_)
    
    #For each value of K, we display score 
    for k in [2,9,16,23,40]:
        counter = 0
        for test in test_data_ : 
            if KNearestNeighbors(k,training_data_,test[0]).finalPrediction() == test[1]:
                counter+=1
        print ("Score : for k = ",k, " is : ",counter/num_test_data)
        
        
    # Plot for data visualize
    # Plot Setosa Points
    X_0 = []
    Y_0 = []
    #Plot Verginica Points
    X_1 = []
    Y_1 = []
    #Plot Versicolor Points
    X_2 = []
    Y_2 = []
    
    for _,element in enumerate(data):  
       if element[1] == 0 :
              X_0.append(element[0][0])
              Y_0.append(element[0][1])
       elif element[1] == 1 :
              X_1.append(element[0][0])
              Y_1.append(element[0][1])
       elif element[1] == 2 :
              X_2.append(element[0][0])
              Y_2.append(element[0][1])              
               
    plt.scatter(X_0,Y_0,color='g',label="setosa")
    plt.scatter(X_1,Y_1,color='r',label="Verginica")
    plt.scatter(X_2,Y_2,color='b',label="Versicolor")
    plt.legend()
    plt.show()


if __name__=="__main__":
	#process_classification_by_custom_class()
    process_classification_by_sickit_learn()
