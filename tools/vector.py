#! /usr/sys/env python 3

import math

#sum of two vectors
def vector_add(v,w):
	return [v_i + w_i for v_i,w_i in zip (v,w)]


#substract two vectors
def vector_substract(v,w):
	return [v_i - w_i for v_i,w_i in zip(v,w)]


#sum of vectors
def vector_sum(vectors):
	result = vectors[0]
	for vect in vectors[1:]:
		result = vector_add(result,vect)
	return vect

#multiply vector and scalar
def scalar_multiply(a,v):
	return [a*v_i for v_i in v]


#mean of vector list
def mean_vector(vectors):
	n = len(vectors)
	return scalar_multiply(1/n,vector_sum(vectors)) 

#dot product of two vectors
def dot(v,w):
	return sum(v_i * w_i for v_i,w_i in zip(v,w))


#sum of squares
def sum_of_squares(v):
	return dot(v,v)


def magnitude(v):
	return math.sqrt(sum_of_squares(v))

def squared_distance(v,w):
	return sum_of_squares(vector_substract(v,w))

def distance(v,w):
	return math.sqrt(squared_distance(v,w))
	
