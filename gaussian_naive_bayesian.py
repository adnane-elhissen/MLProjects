from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB 
from sklearn import metrics 

 
# load the iris dataset 
iris = load_iris() 

# store labeled data from dataset  
X = iris.data 
y = iris.target 
  
# splitting X and y into training and testing sets  
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=1) 
  
# training the model for testing
guassian_distribution = GaussianNB() 
guassian_distribution.fit(X_train, y_train) 
  
# making prediction on testing data 
y_pred = guassian_distribution.predict(X_test) 
  
# comparing y_red and y_test for the score 
print("Gaussian Model score (in %):", metrics.accuracy_score(y_test, y_pred)*100)
