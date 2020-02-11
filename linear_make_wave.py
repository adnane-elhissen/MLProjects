from sklearn.linear_model import LinearRegression
import mglearn
from sklearn.model_selection import train_test_split


from linear_regression import CustomLinearRegression 
from ridge_penalty import CustomPenaltyRidge

from tools import vector as vect
from tools import centrale_tendencies as ct

from sklearn import datasets, linear_model

"""
def score(X,y,beta):
    y_ = ct.mean(y)
    sum_of_squared_error = sum ((y_i - vect.dot(x_i,beta) - y_) ** 2 for x_i,y_i in zip(X,y))
    mea = sum(v ** 2 for v in ct.de_mean(y))
    return 1.0 - (sum_of_squared_error / mea)
"""

X, y = mglearn.datasets.make_wave(n_samples=60)
#X,y = datasets.load_boston()
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

lr = LinearRegression().fit(X_train, y_train)

#rlr = LinearRegression().fit(X_train, y_train)
s = lr.score(X_train,y_train)

    
cl = CustomLinearRegression()
beta = cl.fit(X_train,y_train)

r = cl.multiple_r_squared(X_test,y_test,beta)

cr = CustomPenaltyRidge(10)
br = cr.fit(X_train, y_train)

cs = cr.multiple_r_squared(X_test,y_test,br)

from sklearn.linear_model import Ridge

ridge = Ridge().fit(X_train, y_train)
print("Training set score: {:.2f}".format(ridge.score(X_train, y_train)))
print("Test set score: {:.2f}".format(ridge.score(X_test, y_test)))




