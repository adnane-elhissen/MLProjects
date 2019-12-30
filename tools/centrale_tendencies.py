#! /usr/sys/env python 3

from collections import Counter
from . import vector as vect
import math



def mean(x):
	return sum(x)/len(x)

def median(v):

	n = len(v)
	sorted_v = sorted(v)
	if n%2 == 1:
		return sorted_v[n//2]
	else :
		midpoint = n//2
		return (sorted_v[midpoint] - sorted_v[midpoint -1]) / 2


def quantile(v,p):
	index = int(p*len(v))
	return sorted(v)[index]

def mode(v):
	dict_count = Counter(v)
	max_counts = max(dict_count.values())
	return [x_i for x_i,val in dict_count.iteritems() if val == max_counts]

def de_mean(x):
	xbar = mean(x)
	return [x_i - xbar for x_i in x ]

def variance(x):
	n = len(x)
	x_bar = de_mean(x)
	return vect.sum_of_squares(x_bar) / (n-1)

def standard_deviation(x):
	return math.sqrt(variance(x))

def interquartile_range(x):
	return quantile(x,0.75)-quantile(x,0.25)

def covariance(x,y):
	n = len(x)
	return dot(de_mean(x),de_mean(y)) /(n-1)

def correlation(x,y):
	std_x = standard_deviation(x)
	std_y = standard_deviation(y)
	if std_x>0 and std_y>0:
		return covariance(x,y)/(std_x * std_y)
	else :
		return 0



