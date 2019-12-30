#! /usr/sys/env python3


from tools import vector as vect
from collections import Counter


# The purpose of KNearestNeighbors Class is to classify point basing on calculating distance approach 
# And Predict cLassification of any input point 
   
class KNearestNeighbors:
    
    def __init__(self,k,labeled_points,point):
        self.k_parameter = k
        self.training_points = labeled_points
        self.point_checked = point
        self.sorted_points = []
        self.point_size = len(point)
        
    def sortedPointsByDistance(self):
            self.sorted_points = sorted(self.training_points,key =lambda point:vect.distance(point[0],self.point_checked))
            
            
    def majorityVote (self,labeled_list):
        count = Counter(labeled_list)
        winner,winner_count = count.most_common(1)[0]
        num_count = len ([c for c in count.values() if c==winner_count])
        if num_count==1:
            return winner
        else :
            return self.majorityVote (labeled_list[:-1])
            
    def finalPrediction(self):
         self.sortedPointsByDistance()
         k_nearest = [label for _,label in self.sorted_points[:self.k_parameter]]
         final = self.majorityVote (k_nearest)
         return final
   


        
    
