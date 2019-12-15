#! /usr/bin/env python3

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import math
from collections import Counter

def distance (p0,p1):
	return (math.sqrt ( ((p0[0]-p1[0])**2) + ((p0[1]-p1[1])**2) ) )	

def split_data(data,num_train):
	results = [],[]
	i=0
	for row in data : 
		if (i<num_train):
			results [0].append(row)
		else:
			results [1].append(row)
		i+=1
	return results

def train_test_split (x,y,z,num_train):
	data = zip(x,y,z)
	train,test = split_data(data,num_train)
	x_train,y_train,z_train = zip(*train)
	x_test,y_test,z_test = zip(*test)
	return x_train,x_test,y_train,y_test,z_train,z_test

def majority_vote (labeled_list):
	count = Counter(labeled_list)
	winner,winner_count = count.most_common(1)[0]
	num_count = len ([c for c in count.values() if c==winner_count])
	if num_count==1:
		return winner
	else :
		return majority_vote (labeled_list[:-1])


def knn_classify (k,labeled_points,new_point):
	l_distance = sorted(labeled_points,
				key=lambda point: distance(point,new_point))
	k_nearest = [label for _,_,label in l_distance[:k]]
	return majority_vote (k_nearest)

def score(k,training_data_,test_data_):
	num_test_data = len (test_data_)
	counter = 0
	for data in test_data_ : 
		if knn_classify (k,training_data_,data) == data[2]:
			counter+=1
	return counter/num_test_data



if __name__=="__main__":
	# 0 ==> setosa , 1==> verginica, 2==>versicolor
	iris = pd.read_csv("iris.csv")
	x=iris.loc[:,"petal_length"]
	y=iris.loc[:,"petal_width"]
	z=iris.loc[:,"species"]
	n_train = 0.80 *len(x) 
	x_train,x_test,y_train,y_test,z_train,z_test = train_test_split(x,y,z,n_train)
	data = zip(x,y,z)
	length = 2.5
	width = 0.75
	training_data=zip(x_train,y_train,z_train)
	test_data = zip(x_test,y_test,z_test)

	test_data_ = list(test_data)
	training_data_ = list(training_data)

	point  = (length,width)
	winner = knn_classify (3,data,point)

	k = [2,9,16,23,40]
	for i in k :
		print ("-------------",i,"------")
		print ("score of k-NN :",score(i,training_data_,test_data_))
		#print ("print test_data_",list(test_data))
	plt.scatter(x[z==0],y[z==0],color='g',label="setosa")
	plt.scatter(x[z==1],y[z==1],color='r',label="verginica")
	plt.scatter(x[z==2],y[z==2],color='b',label="versicolor")
	plt.scatter(length,width,color='k')
	plt.legend()
	plt.show()

	
