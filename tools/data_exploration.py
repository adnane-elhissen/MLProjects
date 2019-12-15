#! /usr/sys/env python 3


import math
from collections import Counter,defaultdict
import matplotlib.pyplot as plt
import random
from probabilities import *
from functools import partial



def bucketsize(point, bucket_size):
	return bucket_size * math.floor(point/bucket_size)


def make_histogram(points,bucket_size):
	return Counter(bucketsize(point,bucket_size) for point in points)

def plot_histogram(points,bucket_size,title=""):
	histogram = make_histogram(points,bucket_size)	
	plt.bar(histogram.keys(),histogram.values(),width=bucket_size)
	plt.title(title)
	plt.show()
"""
random.seed(0)
uniform = [200*random.random() -100 for _ in range(10000)]

normal = [57 * inverse_normal_cdf(random.random()) for _ in range(10)]

#plot_histogram(uniform,10,"Uniform Histogram")

#plot_histogram(normal,10,"normal Histogram")
"""
def random_normal():
	return inverse_normal_cdf(random.random())

def correlation_matrix(data):
	_,num_columns = shape(data)
	def matrix_entry(i,j):
		return correlation(get_column(data,i),get_column(data,j))
	return make_matrix(num_columns, num_columns, matrix_entry)

def group_by(grouper,rows,value_transform=None):
	grouped = defaultdict(list)
	for row in rows:
		grouped[grouper(row)].append(row)

	if value_transform is None:
		return grouped
	else : 
		return { key : value_transform(rows) for key,rows in grouped.iteritems()}

def scale(data_matrix):
	num_rows, num_cols = shape(data_matrix)
	means = [mean(get_column(data_matrix,j)) for j in range(num_cols)]
	stdevs = [standard_deviation(get_column(data_matrix,j)) for j in range(num_cols)]
	return means,stdevs

def rescale(data_matrix):
	means,stdevs = scale(data_matrix)
	def rescaled(i,j):
		if stdevs[j]>0:
			return (data_matrix[i][j] - means[j]) / stdevs[j]
		else : 
			return data_matrix[i][j]


	num_rows,num_cols = shape(data_matrix)
	return make_matrix(num_rows,num_cols,rescaled)	


## reduce dimension

def de_mean_matrix(A):
	nr,nc = shape(A)
	column_means,_ = scale(A)
	return make_matrix(nr,nc,lambda i,j:A[i][j]-column_means[j])

def direction(w):
	mag = magnitude(w)
	return [w_i/mag for w_i in w]

def directional_variance_i(x_i,w):
	return dot(x_i,direction(w))**2

def directional_variance(X,w):
	return sum(directional_variance_i(x_i,w) for x_i in X)



def directional_variance_gradient_i(x_i,w):
	projection_length = dot(x_i,direction(w))
	return [2*projection_length*x_ij for x_ij in x_i]

def directional_variance_gradient(X,w):
	return vector_sum(directional_variance_gradient_i(x_i,w) for x_i in X)


def first_principal_component(X):
	guess = [1 for _ in X[0]]
	unscaled_maximizer = maximize_batch(
		partial(directional_variance,X),
		partial(directional_variance_gradient,X),
		guess)
	return direction(unscaled_maximizer)

def project(v,w):
	projection_length = dot(v,w)
	return scalar_multiply(projection_length,w)

def remove_projection_from_vector(v,w):
	return vector_substract(v,project(v,w))

def remove_projection(X,w):
	return [ remove_projection_from_vector(x_i,w) for x_i in X]


def principal_component_analysis(X,num_components):
	components = []
	for _ in range(num_components):
		component = first_principal_component(X)
		components.append(component)
		X = remove_projection(X,component)
	return components


def transform_vector (v,components):
	return [dot(v,w) for w in components]

def transform (X,components):
	return [transform_vector(x_i,components) for x_i in X]




def split_data(data,prob):
     results = [],[]
     for row in data:
         results[0 if random.random() < prob else 1].append(row)
     return results	

def train_test_split(x,y,test_pct):
	data = zip(x,y)
	train,test = split_data(data,1-test_pct)
	X_train,y_train = zip(*train)
	X_test,y_test = zip(*test)
	return X_train,X_test,y_train,y_test

def accuracy(vp,fp,fn,vn):
	correct = vp+vn
	total = vp + fp + fn + vn
	return correct/total

def precision (vp,fp,fn,vn):
	return vp /(vp+fp)
	
def recall (vp,fp,fn,vn):
	return vp /(vp+fn)

def f1_score (vp,fp,fn,vn):
	p = precision (vp,fp,fn,vn)
	r = recall (vp,fp,fn,vn)
	return (2*p*r) / (p+r)






















	




