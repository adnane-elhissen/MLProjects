#! /usr/sys/env python 3

from __future__ import absolute_import
import math
from collections import Counter,defaultdict
import matplotlib.pyplot as plt
import random
from tools import centrale_tendencies as ct
from tools import vector as vect
from tools import descentGradient as dg
from tools import matrix as mx
from tools import probabilities as pb
from functools import partial
import csv 

def load_data_set(filename):
    with open(filename,'r')as f:
        return list(csv.reader(f,delimiter=','))

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
	return pb.inverse_normal_cdf(random.random())

def correlation_matrix(data):
	_,num_columns = mx.shape(data)
	def matrix_entry(i,j):
		return ct.correlation(mx.get_column(data,i),mx.get_column(data,j))
	return mx.make_matrix(num_columns, num_columns, matrix_entry)

def group_by(grouper,rows,value_transform=None):
	grouped = defaultdict(list)
	for row in rows:
		grouped[grouper(row)].append(row)

	if value_transform is None:
		return grouped
	else : 
		return { key : value_transform(rows) for key,rows in grouped.iteritems()}

def scale(data_matrix):
	num_rows, num_cols = mx.shape(data_matrix)
	means = [ct.mean(mx.get_column(data_matrix,j)) for j in range(num_cols)]
	stdevs = [ct.standard_deviation(mx.get_column(data_matrix,j)) for j in range(num_cols)]
	return means,stdevs

def rescale(data_matrix):
	means,stdevs = scale(data_matrix)
	def rescaled(i,j):
		if stdevs[j]>0:
			return (data_matrix[i][j] - means[j]) / stdevs[j]
		else : 
			return data_matrix[i][j]


	num_rows,num_cols = mx.shape(data_matrix)
	return mx.make_matrix(num_rows,num_cols,rescaled)	


## reduce dimension

def de_mean_matrix(A):
	nr,nc = mx.shape(A)
	column_means,_ = scale(A)
	return mx.make_matrix(nr,nc,lambda i,j:A[i][j]-column_means[j])

def direction(w):
	mag = vect.magnitude(w)
	return [w_i/mag for w_i in w]

def directional_variance_i(x_i,w):
	return vect.dot(x_i,direction(w))**2

def directional_variance(X,w):
	return sum(directional_variance_i(x_i,w) for x_i in X)



def directional_variance_gradient_i(x_i,w):
	projection_length = vect.dot(x_i,direction(w))
	return [2*projection_length*x_ij for x_ij in x_i]

def directional_variance_gradient(X,w):
	return vect.vector_sum(directional_variance_gradient_i(x_i,w) for x_i in X)


def first_principal_component(X):
	guess = [1 for _ in X[0]]
	unscaled_maximizer = dg.maximize_batch(
		partial(directional_variance,X),
		partial(directional_variance_gradient,X),
		guess)
	return direction(unscaled_maximizer)

def project(v,w):
	projection_length = vect.dot(v,w)
	return vect.scalar_multiply(projection_length,w)

def remove_projection_from_vector(v,w):
	return vect.vector_substract(v,project(v,w))

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
	return [vect.dot(v,w) for w in components]

def transform (X,components):
	return [transform_vector(x_i,components) for x_i in X]




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

def train_test_split(x,y,test_pct):
	data = zip(x,y)
	train,test = split_data(data,test_pct)
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






















	




