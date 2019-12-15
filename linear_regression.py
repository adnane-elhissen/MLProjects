#! /usr/bin/env python3

import math
import numpy
import matplotlib.pyplot as plt
from functools import partial
import random

###########descente du gradient#########################
def distance (u,v):
	return math.sqrt( sum ( (u_i-v_i)**2 for u_i,v_i in zip(u,v) ) )
 
def sum_squares(v):
	return sum(v_i**2 for v_i in v)

def difference_quotient(f,x,h):
	return (f(x+h) - f(x))/h

def square(x):
	return x**2

def derivative(x):
	return 2*x

def vector_subtract(v,w):
	return [v_i + w_i for v_i,w_i in zip(v,w)]

def scalar_multiply(c,v):
	return [c*v_i for v_i in v]

derivative_estimate = partial (difference_quotient,square,h=0.00001)
x=range(-10,10)
plt.title("derivées réelles Vs dérivées avec la limitte")
der_coord_y = list(map(derivative,x))
est_coord_y = list(map(derivative_estimate,x))
plt.plot(x,der_coord_y,'rx',label='Actual')
plt.plot(x,est_coord_y,'b+',label='Estimate')
plt.legend()
plt.show()

def partial_difference_quotient(f,v,i,h):
	w = [v_i +(h if i==j else 0) for j,v_i in enumerate(v)]
	return (f(w) - f(v)) /h 

def estimate_gradient(f,v,h=0.00001):
	return [partial_difference_quotient(f,v,i,h) for i,_ in enumerate(v)]

def step(v,direction,step_size):
	return [v_i + step_size * direction_i for v_i,direction_i in zip(v,direction)]

def sum_of_squares_gradient(v):
	return[2*v_i for v_i in v]
'''
v = [random.randint(-10,10) for i in range(3)]
tolerance = 0.0000001
c=0
while True:
	gradient = sum_of_squares_gradient(v)
	next_v = step(v,gradient,-0.01)
	if distance (next_v,v) < tolerance:
		break
	v = next_v
	c+=1
print(v)
print(c)
'''
def safe(f):
	def safe_f(*args,**kwargs):
		try:
			return f(*args,**kwargs)
		except:
			return float('inf')
	return safe_f

def minimize_batch(target_fn,gradient_fn,theta_0,tolerance=0.000001):
	step_sizes = [100,10,1,0.1,0.01,0.001,0.0001,0.00001]
	theta = theta_0
	target_fn = safe(target_fn)
	value = target_fn(theta)
	while True:
		gradient = gradient_fn(theta)
		next_thetas = [step(theta,gradient,-step_size) for step_size in step_sizes]
		next_theta = min(next_thetas,key=target_fn)
		next_value = target_fn(next_theta)
		if(abs(value - next_value) < tolerance)
			return theta
		else:
			theta,value = next_theta,next_value

def negate(f):
	return lambda *args,**kwargs: -f(*args,**kwargs)

def negate_all(f):
	return lambda *args,**kwargs : [-y for y in f(*args,**kwargs)]

def maximaze_batch(target_fn,gradient_fn,theta_0,tolerance=0.000001):
	return minimize_batch(negate(target_fn),negate_all(gradient_fn),theta_0,tolerance=0.000001)

def in_random_order(data):
	indexes = [i for i,_ in enumerate(data)]
	random.shuffle (indexes)
	for i in indexes:
		yield data[i]
def minimize_stochastic(target_fn,gradient_fn,x,y,theta_0,alpha_0=0.001):
	data = zip(x,y)
	theta = theta_0
	alpha = alpha_0
	min_theta,min_value = None,float("inf")
	iterations_with_no_improvement = 0

	while iterations_with_no_improvement < 100:
		value = ( sum(target_fn,x_i,y_i,theta) for x_i,y_i in data )
		if value < min_value : 
			min_theta,min_value = theta,value
			iterations_with_no_improvement = 0
			alpha = alpha_0
		else : 
			iterations_with_no_improvement +=1
			alpha *= 0.9
			for x_i,y_i in in_random_order(data):
				gradient_i = gradient_fn(x_i,y_i,theta)
				theta = vector_substract (theta,scalar_multiply(alpha,gradient_i))

	return min_theta		
########################################################



def de_mean(x):
	x_bar = mean(x)
	return [x_i - x_bar for x_i in x]

def dot (x,y):
	return sum (x_i*y_i for x_i,y_i in zip(x,y))

def sum_of_squared (x):
	return sum (x_i*x_i for x_i in x)

def variance (x):
	n = len(x)
	return sum(diff**2 for diff in de_mean(x))/n

def standard_deviation(x):
	return math.sqrt(variance(x))

def covariance (x,y):
	n = len(x)
	return dot(de_mean(x),de_mean(y))/n

def correlation (x,y):
	std_x = standard_deviation(x)
	std_y = standard_deviation(y)
	if std_x > 0 and std_y > 0:
		return covariance(x,y)/(std_x*std_y)
	else : 
		return 0
#########regression simple##########################
'''
def least_squares_fit(x,y):
	beta = correlation(x,y) * standard_deviation(y) /standard_deviation(x)
	alpha = mean(y) - beta*mean(x)
	return alpha,beta

def predict(alpha,beta,x_i):
	return beta*x_i + alpha

def error(alpha,beta,x_i,y_i):
	return y_i - predict(alpha,beta,x_i)

def sum_of_squared_errors(alpha,beta,x,y):
	return sum(error(alpha,beta,x_i,y_i)**2 for x_i,y_i in zip(x,y))

def total_sum_of_squares(y):
	return sum(v ** 2 for v in de_mean(y))

def r_squared(alpha,beta,x,y):
	return 1.0 - (sum_of_squared_errors(alpha,beta,x,y)/total_sum_of_squares(y)) 

def squared_error(x_i,y_i,theta):
	alpha,beta = theta
	return error(alpha,beta,x_i,y_i)**2
def squared_error_gradient(x_i,y_i,theta):
	alpha,beta = theta
	return [-2 * error(alpha,beta,x_i,y_i),-2 * error(alpha,beta,x_i,y_i) * x_i]

random.seed(0)
theta = [random.random(),random.random()]
alpha,beta = minimize_stochastic(squared_error,squared_error_gradient,num_friends_good,daily_minutes_good,theta,0.0001)
print (alpha,beta)
'''
#######################################################




################regression multiple####################
def total_sum_of_squares(y):
	return sum(v ** 2 for v in de_mean(y))

def predict(x_i,beta):
	return dot(x_i,beta)

def error(x_i,y_i,beta):
	return y_i - predict(x_i,beta)

def squared_error(x_i,y_i,beta):
	return error(x_i,y_i,beta) ** 2

def squared_error_gradient(x_i,y_i,beta):
	return [-2 * x_ij * error(x_i,y_i,beta) for x_ij in x_i]

def estimate_beta(x,y):
	beta_initial = [random.random() for x_i in x[0]]
	return minimize_stochastic(squared_error,squared_error_gradient,x,y,beta_initial,0.001)

random.seed(0)
beta = estimate_beta(x,daily_minutes_good)


def multiple_r_squared(x,y,beta):
	sum_of_squared_errors = sum (error (x_i,y_i,beta) ** 2 for x_i,y_i in zip (x,y) )
	return 1.0 - sum_of_squared_errors/total_sum_of_squares(y)


def bootstrap_sample(data):
	return [random.choice(data) for _ in data]

def bootstrap_statistic(data,stats_fn,num_samples):
	return [stats_fn(bootstrap_sample(data)) for _ in range(num_samples)]

def ridge_penalty(beta,alpha)
	return alpha * dot(beta[1:],beta[1:])

def squared_error_ridge(x_i,y_i,beta,alpha):
	return error(x_i,y_i,beta) ** 2 + ridge_penalty(beta,alpha)

def ridge_penalty_gradient (beta,alpha):
	return [2*alpha*beta_i for beta_i in beta[1:]]

def squared_error_ridge_gradient(x_i,y_i,beta,alpha):
	return vector_add(squared_error_gradient(x_i,y_i,beta),ridge_penalty_gradient(beta,alpha))

def estimate_beta_ridge(x,y,alpha):
	beta_initial = [random.random() for _ in x[0]]
	return minimize_stochastic (partial(squared_error_ridge,alpha=alpha),partial(squared_error_ridge,alpha=alpha),x,y,beta_initial,0.001)

def lasso_penalty(beta,alpha):
	return alpha * sum(abs(beta_i) for beta_i in beta[1:])

random.seed(0)
beta_0_01 = estimate_beta_ridge(x,daily_minutes_good,alpha=0.001)
multiple_r_squared(x,daily_minutes_good,beta_0_01)



































	
