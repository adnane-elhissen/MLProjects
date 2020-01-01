#! /usr/bin/env python3
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

#Convert words from Upper case to Lower Case
def tokenize(message):
	message = message.lower()
	all_words = re.findall("[a-z0-9']+",message)
	return set(all_words)

# Recover data set and return Dictionary of words and their occurance in spam and no spam messages 
def count_words(training_set):
	counts = defaultdict(lambda:[0,0])
	for message,is_spam in training_set:
		for word in tokenize(message):
			counts[word][0 if is_spam else 1] +=1
	return counts

#Calculate probability of each words in dictionary returned by count_words function
def word_probabilities(counts,total_spams,total_non_spams,k):
	return [(w,(spam+k)/(total_spams +2*k),(non_spam+k)/(total_non_spams +2*k))
		for w,(spam,non_spam) in counts.items()]



# P(Xi | S) = (k + number of spam messages containing wi) / (2k + number of spam messages)
# wi ∈ Xi and Xi is set of words , k random compteur to avoid a zero value of probability P(wi | S) = 0     


#Define probability of message to be spam or not based on probability of each word in dictionary
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
    # data sets of spam 
    # "ham" means that we use a no spam message
	path = r"data_sets/spam/*/*"
    # We store labeled data 
	data = []
	for fn in glob.glob(path):
		is_spam ="ham" not in fn 
		with open(fn,encoding="utf8", errors='ignore') as fl:
			for line in fl :
				if line.startswith("Subject:"):
					subject = re.sub(r"^Subject: ","",line).strip()
					data.append((subject,is_spam))
			
			
	random.seed(0)
    #Split data into training and test
	train_data,test_data =split_data(data,0.75)
    
    #Traning models based on NaiveBayesClassifier
	classifier = NaiveBayesClassifier()
	classifier.train(train_data)
	classified = [(subject,is_spam,classifier.classify(subject)) for subject,is_spam in test_data]
    
    #We suppose that spam_probability > 0.5 corresponds to spam message 
	counts = Counter((is_spam,spam_probability>0.5) for _,is_spam,spam_probability in classified)







	
