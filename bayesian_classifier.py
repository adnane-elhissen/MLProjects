#! /usr/bin/env python2
# -*- coding: utf-8 -*-

import math
import re ,glob
import random
from collections import defaultdict
from collections import Counter

def split_data(data,num_train):
	results = [],[]
	i=0
	for row in data : 
		if (i<num_train):
			results [0].append(row)
		else:
			results [1].append(row)
		i+=1
	return results

def train_test_split (x,y,z,num_train):
	data = zip(x,y,z)
	train,test = split_data(data,num_train)
	x_train,y_train,z_train = zip(*train)
	x_test,y_test,z_test = zip(*test)
	return x_train,x_test,y_train,y_test,z_train,z_test

def tokenize(message):
	message = message.lower()
	all_words = re.findall("[a-z0-9']+",message)
	return set(all_words)

def count_words(training_set):
	counts = defaultdict(lambda:[0,0])
	for message,is_spam in training_set:
		for word in tokenize(message):
			counts[word][0 if is_spam else 1] +=1
	return counts

def word_probabilities(counts,total_spams,total_non_spams,k):
	return [(w,(spam+k)/(total_spams +2*k),(non_spam+k)/(total_non_spams +2*k))
		for w,(spam,non_spam) in counts.items()]

def spam_probability(word_probs,message):
	message_words = tokenize(message)
	log_prob_if_spam = log_prob_if_not_spam = 0.0
	for word,prob_if_spam,prob_if_not_spam in word_probs:
		if word in message_words:
			log_prob_if_spam += math.log (prob_if_spam)
			log_prob_if_not_spam += math.log (prob_if_not_spam)

		else:
			log_prob_if_spam += math.log (1.0 - prob_if_spam)
			log_prob_if_not_spam += math.log (1.0 - prob_if_not_spam)

	prob_if_spam = math.exp(log_prob_if_spam)
	prob_if_not_spam = math.exp(log_prob_if_not_spam)
	return prob_if_spam/(prob_if_spam + prob_if_not_spam)

class NaiveBayesClassifier:
	def init(self,a=0.5):
		self.k = a
		self.word_probs= []
	def train(self,training_set):
		num_spams = len ([is_spam for message,is_spam in training_set if is_spam])
		num_non_spams = len(training_set) - num_spams
		word_counts = count_words(training_set)
		self.word_probs = word_probabilities (word_counts,num_spams,num_non_spams,0.5)
	def classify(self,message):
		return spam_probability(self.word_probs,message)


if __name__=="__main__":
	path = r"/home/adnane/Bureau/projet_data/spam/*/*"
	data = []
	for fn in glob.glob(path):
		is_spam ="ham" not in fn 
		with open(fn,encoding="utf8", errors='ignore') as fl:
			for line in fl :
				#print (line)
				if line.startswith("Subject:"):
					subject = re.sub(r"^Subject: ","",line).strip()
					data.append((subject,is_spam))
			
			
	random.seed(0)
	train_data,test_data =split_data(data,0.75)
	classifier = NaiveBayesClassifier()
	classifier.train(train_data)
	classified = [(subject,is_spam,classifier.classify(subject)) for subject,is_spam in test_data]
	counts = Counter((is_spam,spam_probability>0.5) for _,is_spam,spam_probability in classified)
	print(classifier.word_probs)






	
