#! /usr/bin/env python3

from collections import Counter
import math

def dot(v,w):
	return sum (v_i*w_i for v_i,w_i in zip(v,w))

def vector_subtract(v,w):
	return [v_i-w_i for v_i,w_i in zip(v,w)]

def sum_of_squares(v):
	return dot(v,v)

def magnitude(v):
	return math.sqrt(sum_of_squares(v))

def distance (v,w):
	return magnitude(vector_subtract(v,w))


def split_data(data,prob):
     results = [],[]
     for row in data:
         results[0 if random.random() < prob else 1].append(row)
     return results	

def train_test_split(x,y,test_pct):
	data = zip(x,y)
	train,test = split_data(data,1-test_pct)
	X_train,y_train = zip(*train)
	X_test,y_test = zip(*test)
	return X_train,X_test,y_train,y_test

def precision (vp,fp,fn,vn):
	return vp /(vp+fp)
	
def recall (vp,fp,fn,vn):
	return vp /(vp+fn)

def f1_score (vp,fp,fn,vn):
	p = precision (vp,fp,fn,vn)
	r = recall (vp,fp,fn,vn)
	return (2*p*r) / (p+r)

# for Knn classifier
def majority_vote(labels):
	vote_counts = Counter(labels)
	winner,winner_count = vote_counts.most_common(1)[0]
	num_winners = len([count for count in vote_counts.values() if count==winner_count])
	if num_winners == 1:
		return winner
	else : 
		return majority_vote(labels[:-1])

def knn_classify(k,labeled_points,new_point):
	#by_distance = [distance (point,new_point) for point,_ in labeled_points]	
	by_distance = sorted(labeled_points,key=lambda point: distance(point[0],new_point))
	#by_distance = sorted(by_distance)
	#print (by_distance)
	k_nearest_labels = [label for _,label in by_distance[:k]]
	return majority_vote(k_nearest_labels)




if __name__=="__main__":

	#x = ["adnane","moi","france"]
	#y = [1,2,3]
	#m = zip (x,y)
	#r,l = split_data(m,0.63)
	#print(r)
	cities = [([-122.3,47.53],"Python"),([-96.85,32.85],"Java"),([-89.33,43.13],"R")]


	for k in [1,3,5,7]:
		num_correct = 0
		#print ("je suis ici")
		for city in cities:
			location,actual_language = city
			other_cities = [other_city
					for other_city in cities
					if other_city != city]
			predicted_language = knn_classify(k,other_cities,location)
			#print (predicted_language)
			if predicted_language == actual_language:
				num_correct += 1
		print (k,"neighbor[s]:",num_correct,"correct out of",len(cities))











