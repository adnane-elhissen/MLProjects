import matplotlib.pyplot as plt
from functools import partial
import random

from tools import descentGradient as dg
from tools import vector as vect

from linear_regression import CustomLinearRegression

class CustomPenaltyRidge:
    
    def __init__(self,alpha):
        self.linearPart = CustomLinearRegression()
        self.alpha = alpha
    def ridge_penalty(self,beta,alpha):
        return alpha * vect.dot(beta[1:],beta[1:])

    def squared_error_ridge(self,x_i,y_i,beta,alpha):
        	return self.linearPart.error(x_i,y_i,beta) ** 2 + self.ridge_penalty(beta,alpha)
    
    def ridge_penalty_gradient (self,beta,alpha):
        	return [0] + [ 2 * alpha * beta_i for beta_i in beta[1:]]
    
    def squared_error_ridge_gradient(self,x_i,y_i,beta,alpha):
        	return vect.vector_add(self.linearPart.squared_error_gradient(x_i,y_i,beta),self.ridge_penalty_gradient(beta,alpha))
    
    def estimate_beta_ridge(self,x,y,alpha):
        beta_initial = [random.random() for _ in x[0]]
        return dg.minimize_stochastic (partial(self.squared_error_ridge,alpha=alpha),partial(self.squared_error_ridge_gradient,alpha=alpha),x,y,beta_initial,0.001)
    
    def estimate_sample_beta(self,sample):
        x_sample,y_sample = zip(*sample)
        return self.estimate_beta_ridge(x_sample,y_sample,self.alpha)
    
    def multiple_r_squared(self,x,y,beta):
        sum_of_squared_error = sum (self.squared_error_ridge(x_i,y_i,beta,self.alpha) for x_i,y_i in zip(x,y))
        return 1.0 - (sum_of_squared_error / self.linearPart.total_sum_of_squares(y))
    