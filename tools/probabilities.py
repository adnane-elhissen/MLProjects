#! /usr/sys/env python 3

import math
import matplotlib.pyplot as plt

def uniform_pdf(x):
	return 1 if x>=0 and x<1 else 0

def uniform_cdf(x):
	if x<0 : return 0
	elif x<1 : return x
	else : return 1



def normal_pdf(x,mu=0,sigma=0):
	sqrt_two_pi = math.sqrt(2*math.pi)
	return (1/(sqrt_two_pi*sigma))*(math.exp(-((x-mu)/sigma )**2) /2)



def plot_normal_pdf(x,mu=0,sigma=0):
	x = [x_i for x_i in range (-5,5)]
	plt.plot(x,[normal_pdf(x_i,sigma=1) for x_i in x],'-',label ='mu=0,sigma=1')
	plt.legend()
	plt.title("Distribution normale")
	plt.show() 
	
#cumulative distribution function
def normal_cdf(x,mu=0,sigma=1):	
	return (1 + math.erf((x-mu) / math.sqrt(2) / sigma)) /2

def inverse_normal_cdf(p,mu=0,sigma=1,tolerance=0.00001):
	if mu != 0 or sigma != 1:
		return mu*sigma*inverse_normal_cdf(p,tolerance=tolerance)
	
	low_z = -10.0
	hi_z = 10.0
	mid_z = 0.0
	while hi_z-low_z > tolerance :
		print ("je suis ici")
		mid_z = (low_z-hi_z) / 2
		mid_p = normal_cdf(mid_z)
		if mid_p < p :
			low_z = mid_z
		elif mid_p > p :
			hi_z = mid_z
		else :
			break
	return mid_z

#determinate mu and sigma for binomial(n,p)
def normal_approximation_to_binomial(n,p):
	mu = n * p
	sigma = math.sqrt(n*p*(1-p))
	return mu,sigma	

normal_probability_below = normal_cdf

def normal_probability_above(lo,mu=0,sigma=1):
	return 1 - normal_cdf(lo,mu,sigma)

def normal_probability_between(lo,hi,mu=0,sigma=1):
	return normal_cdf(hi,mu,sigma) - normal_cdf(lo,mu,sigma) 

def normal_probability_outside(lo,hi,mu=0,sigma=1):
	return 1 - normal_probability_between(lo,hi,mu,sigma)

def normal_upper_bound(probability,mu=0,sigma=1):
	return inverse_normal_cdf(probability,mu,sigma)

def normal_lower_bound(probability,mu=0,sigma=1):
	return inverse_normal_cdf(1-probability,mu,sigma)

def normal_two_sided_bounds(probability,mu=0,sigma=1):
	tail_probability = (1-probability)/2
	upper_bound = normal_lower_bound(tail_probability,mu,sigma)
	lower_bound = normal_upper_bound(tail_probability,mu,sigma)
	return lower_bound,upper_bound































