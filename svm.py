

import math
import random

from tools import vector as vect
from tools import descentGradient as dg
import data_exploration as de

from functools import reduce
from functools import partial


from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
import mglearn
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split

from matplotlib.pyplot import plot as plt


#Custom class for Supported Vector Machine
#We suppose a binary classification and linear Kernel to simplify calculations
#We have primal optimization problem
"""
 ------       min (||w||^2 ) such that yi(w.xi + b) - 1 >= 0 for i=1..n
 
 But to convert constrained optimization problem to unconstrained problem, we will use lagrange multipliers
 (Dual approach)
 so : ====> 
 
         objective function will be  : 1/2 * min (||w||^2 ) - sum λi(yi(w.xi + b) - 1 ) for i=1..n
         λi : lagrange multiplier 
         By applying gradient : 
             w = sum (λi.yi.xi) , for i=1..n
             sum(λi.yi) = 0 , for i=1..n
             
             when we replace w on objective function : 
                 sum (λi - 1/2 * sum (λiλj.yiyj.xixj)) i,j = 1..n   /!\ to maximize
             
             
             b = 1/Ns * (sum(ys - sum(λm.ym.xm.xs))) s∊S,m∊S, S = Supported Vectors
            Ns  : number of Supported Vectors            
"""
class CustomSvm:
    
    def __init__(self):
        self.b = 0
        self.omega = []
    
    def kernel_function(self,x,y):
        return vect.dot(x,x) * vect.dot(y,y)
    
    def cost_function(self,x,y,beta):
        a = sum(beta_i for beta_i in beta)
        b = 0
        for beta_i,x_i,y_i in zip (beta,x,y):
            for beta_j,x_j,y_j in zip (beta,x,y):
                b_ += beta_i*beta_j*(vect.dot(x_i,x_j))*(y_i*y_j)
                
        b /= 2
        return a-b                
    
    def cost_function_gradient_i(self,x,y,beta):
         b = 0
         grad_i = []
         i = j = 0
         for beta_i,x_i,y_i in zip (beta,x,y):
            for beta_j,x_j,y_j in zip (beta,x,y):
                if i == j:
                    b += beta_i*(vect.dot(x_i,x_j))*(y_i*y_j)
                else : 
                    b += beta_j*(vect.dot(x_i,x_j))*(y_i*y_j) / 2.0
                j += 1
            b = 0    
            i += 1
            j = 0 
            grad_i.append(1.0 - b)
         return grad_i
    
    def lagrange_multipliers(self,X_train,y_train):
        f = partial(self.cost_function,X_train,y_train)
        gradient_f = partial(self.cost_function_gradient_i,X_train,y_train)
        size = len(X_train[0])
        beta_0 = [random.random() for _ in range(size)]
        return dg.maximize_batch(f,gradient_f,beta_0)
    
    def fit(self,X_train,y_train):
        beta = self.lagrange_multipliers(X_train,y_train)
        omega = reduce(vect.vector_add,[vect.scalar_multiply(beta_i*y_i,x_i) for beta_i,y_i,x_i in zip(beta,y_train,X_train)])
        b = 0
        Ns = 0
        for x_i,y_i in zip(X_train,y_train) :
            if y_i == 1 :
                Ns += 1
                k = 0
                for x_m,y_m,beta_m in zip(X_train,y_train,beta) :
                    if y_m == 1 :
                        k += vect.dot(vect.scalar_multiply(beta_m*y_m,x_m),x_i)
                    
                b += y_i - k
        b /= Ns
        self.b = b
        self.omega = omega
        
    def predict(self,x_i):
        return sum(vect.dot(x_i,self.omega))




        