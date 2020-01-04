import matplotlib.pyplot as plt
from functools import partial
import random

from tools import descentGradient as dg
from tools import vector as vect

from linear_regression import CustomLinearRegression

class CustomPenaltyLasso:
    
    def __init__(self,alpha):
        self.alpha = alpha
        self.linearPart = CustomLinearRegression()
        
    def lasso_penalty(self,beta,alpha):
        return alpha * sum(abs(beta_i) for beta_i in beta[1:])  

    def squared_error_lasso(self,x_i,y_i,beta,alpha):
        return self.linearPart.error(x_i,y_i,beta) ** 2 + self.lasso_penalty(beta,alpha)

    def lasso_penalty_gradient(self,beta,alpha):
        result = []
        for ele in beta[1:]:
            if ele > 0 :
                result.append(alpha)
            else : 
                result.append(-alpha)            
        return [0] + result

    def squared_error_lasso_gradient(self,x_i,y_i,beta,alpha):
        return vect.vector_add(self.linearPart.squared_error_gradient(x_i,y_i,beta),self.lasso_penalty_gradient(beta,alpha))

    def estimate_beta_lasso(self,x,y,alpha):
        beta_initial = [random.random() for _ in x[0]]
        return dg.minimize_stochastic (partial(self.squared_error_lasso,alpha=alpha),partial(self.squared_error_lasso_gradient,alpha=alpha),x,y,beta_initial,0.001)
    
    def estimate_sample_beta(self,sample):
        x_sample,y_sample = zip(*sample)
        return self.estimate_beta_lasso(x_sample,y_sample,self.alpha)

    def multiple_r_squared(self,x,y,beta):
        sum_of_squared_error = sum (self.squared_error_lasso(x_i,y_i,beta,self.alpha) for x_i,y_i in zip(x,y))
        return 1.0 - (sum_of_squared_error / self.linearPart.total_sum_of_squares(y))        