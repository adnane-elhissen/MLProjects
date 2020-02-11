

import math
import random

from tools import vector as vect
from tools import descentGradient as dg
import data_exploration as de

from functools import reduce
from functools import partial


#logistic function 
def logistic(x):
    return 1.0 / (1.0 + math.exp(-x))

# derivative of logistic function
def logistic_prime(x):
    return logistic(x)*(1.0 - logistic(x))
"""
we should finaly define model as : yi = f(xiβ) +εi ; f = logistic function
f(x) ∈ [0,1], so we can consider the density of probability :
p(yi|xi,β) = ( f(xiβ)^yi ) * (1-f(xiβ))^(1-yi)
In this case, we talk about likelihood function that can be maximized by logarithm function
log( L(β | xi,yi)) = yi*log(f(xiβ)) + (1-yi)*log((1-f(xiβ)))
"""
class CustomLogisticalRegression:
    
    def logistic_log_likelihood_i(self,x_i,y_i,beta):
        if y_i == 1:
            return math.log(logistic(vect.dot(x_i,beta)))
        else:
            return math.log(1 - logistic(vect.dot(x_i,beta)))
    
    def logistic_log_likelihood(self,x,y,beta):
        return sum(self.logistic_log_likelihood_i(x_i,y_i,beta) for x_i,y_i in zip (x,y))
    
    def logistic_log_partial_ij(self,x_i,y_i,beta,j):
        return (y_i - logistic(vect.dot(x_i,beta))) * x_i[j]
    
    def logistic_log_gradient_i(self,x_i,y_i,beta):
        return [self.logistic_log_partial_ij(x_i,y_i,beta,j) for j,_ in enumerate(beta)]
    
    def logistic_log_gradient(self,x,y,beta):
        return reduce(vect.vector_add,[self.logistic_log_gradient_i(x_i,y_i,beta) for x_i,y_i in zip(x,y)])
        
    def fit(self,X_train,y_train):
        f = partial(self.logistic_log_likelihood,X_train,y_train)
        gradient_f = partial(self.logistic_log_gradient,X_train,y_train)
        size = len(X_train[0])
        beta_0 = [random.random() for _ in range(size)]
        return dg.maximize_batch(f,gradient_f,beta_0)

    def score(self,X_test,y_test,beta):
        true_positives = false_positives = true_negatives = false_negatives = 0
        for x_i,y_i in zip(X_test,y_test):
            predict = logistic(vect.dot(x_i,beta))
            
            if y_i == 1 and predict >=0.5:
                true_positives += 1
            elif y_i == 1 :
                false_negatives += 1
            elif predict >=0.5:
                false_positives += 1
            else:
                true_negatives += 1

        return de.f1_score(true_positives,false_positives,false_negatives,true_negatives)                