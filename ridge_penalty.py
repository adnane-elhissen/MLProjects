import matplotlib.pyplot as plt
from functools import partial
import random

from tools import descentGradient as dg
from tools import vector as vect
from tools import centrale_tendencies as ct

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
        beta_initial = [0.4 for _ in x[0]]
        #beta_initial = x[0]
        return dg.minimize_stochastic (partial(self.squared_error_ridge,alpha=alpha),partial(self.squared_error_ridge_gradient,alpha=alpha),x,y,beta_initial,0.0000001)
        
    
    def estimate_sample_beta(self,x,y):
        #x_sample,y_sample = zip(*sample)
        return self.estimate_beta_ridge(x,y,self.alpha)
    
    def multiple_r_squared(self,x,y,beta):
        y_ = ct.mean(y)
        #sum_of_squared_error = sum (self.squared_error_ridge(x_i,y_i,beta,self.alpha) for x_i,y_i in zip(x,y))
        sum_of_squared_error = sum ((self.linearPart.error(x_i,y_i,beta) - y_) ** 2 + self.ridge_penalty(beta,self.alpha) for x_i,y_i in zip(x,y))
        return 1.0 - (sum_of_squared_error / self.linearPart.total_sum_of_squares(y))
    
    def fit(self,x,y):
        return self.estimate_sample_beta(x,y)
    