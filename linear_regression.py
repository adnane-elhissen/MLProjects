#! /usr/bin/env python3

import math
import numpy
import matplotlib.pyplot as plt
from functools import partial
import random

from tools import centrale_tendencies as ct
from tools import descentGradient as dg


from tools import vector as vect
from tools import probabilities as pb

"""

# Basing on simple linear regression as : Yi = beta*Xi + alpha 
# Yi will be used as predicted value using alpha and beta parameters
# Yi ==> predict
def predict(alpha,beta,x_i):
    return beta*x_i + alpha


# Finally, we should choose alpha and beta that will minimize error between reality and prediction
# error = Y_i - Yi = Y_i - predict(alpha,beta,Xi)     
def error(alpha,beta,x_i,y_i):
    return y_i - predict(alpha,beta,x_i)


# Some times, we may have two input x1 with big error and x2 with small error. 
# Therefore, there is a compensation
# it's interesting to calculate squared error as :    
def sum_of_squared_errors(alpha,beta,x,y):
    return sum(error(alpha,beta,x_i,y_i) ** 2 for x_i,y_i in zip (x,y))

# Values of alpha and beta that minimize error are defined by :
# beta = correlation (x,y)* std(y)/std(x)
# alpha = mean(y) - beta*mean(x)
def least_squares_fit(x,y):
    beta = ct.correlation(x,y) * ct.standard_deviation(y) / ct.standard_deviation(x)
    alpha = ct.mean(y) - beta * ct.mean(x)
    return alpha,beta

# For alpha and beta, we can use descent gradient also
def squared_error(x_i,y_i,theta):
    alpha,beta = theta
    return error(alpha,beta,x_i,y_i) ** 2

def gradient_of_squared_error(x_i,y_i,theta):
    alpha,beta = theta
    return [-2 * error(alpha,beta,x_i,y_i), -2 * x_i * error(alpha,beta,x_i,y_i)]

def parameters_with_descent_gradient(X,y):
    random.seed(0)
    theta = [random.random(),random.random()]
    alpha,beta = dg.minimize_stochastic(squared_error,gradient_of_squared_error,X,y,theta,0.0001)
    return alpha,beta

# We should calculate R^2 coefficient : 
     
def total_sum_of_squares(y):
    return sum(v ** 2 for v in ct.de_mean(y))

def r_squared (alpha,beta,x,y):
    return 1.0 - ( sum_of_squared_errors(alpha,beta,x,y) / total_sum_of_squares(y) )

"""
######################################################################
class CustomLinearRegression :

    def total_sum_of_squares(self,y):
        return sum(v ** 2 for v in ct.de_mean(y))

    def predict(self,x_i,beta):
        return vect.dot(x_i,beta)
    
    def error(self,x_i,y_i,beta):
        return y_i - self.predict(x_i,beta)

    def squared_error(self,x_i,y_i,beta):
        return self.error(x_i,y_i,beta) ** 2        

    def squared_error_gradient(self,x_i,y_i,beta):
        return [- 2 * x_ij * self.error(x_i,y_i,beta) for x_ij in x_i]


    def estimate_beta(self,x,y):
        beta_0 = [random.random() for x_i in x[0]]
        return dg.minimize_stochastic(self.squared_error,self.squared_error_gradient,x,y,beta_0,0.0001)
    
    def estimate_sample_beta(self,sample):
        x_sample,y_sample = zip(*sample)
        return self.estimate_beta(x_sample,y_sample)

    def multiple_r_squared(self,x,y,beta):
        sum_of_squared_error = sum (self.squared_error(x_i,y_i,beta) for x_i,y_i in zip(x,y))
        return 1.0 - (sum_of_squared_error / self.total_sum_of_squares(y))










	
