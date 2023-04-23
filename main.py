#from thefuzz import fuzz, process
#import numpy as np
import pandas as pd
#import tensorflow as tf
#import csv
#import json
#from tensorflow import keras


review_path = "devided_dataset_v2/CDs_and_Vinyl/train/review_training.json"
product_path = "devided_dataset_v2/CDs_and_Vinyl/train/product_training.json"
product_data = pd.read_json(product_path)
review_data = pd.read_json(review_path)

#Training and coming up with feature vector
feature_vector = {}  # key = product ; values = array of final feature values
temp_dict = {} # temp dict to compile current feature info for a product accross features

productIDs = product_data[['asin', 'awesomeness']]
length = len(list(productIDs['asin']))
for i in range(length):
    feature_vector[productIDs['asin'][i]] = [productIDs['awesomeness'][i]]
    temp_dict[productIDs['asin'][i]] = [productIDs['awesomeness'][i],[], [], [],[],[],[],[]]
    
#order array indices according to order numbered here; for reature vector

#1 avg positive summary 
#2 avg negative summary
#3 avg positive review text
#4 avg negative review text
#5 avg vote credibility score
#6 image credibility score
#7 verified credibility score
#8 time score

#data collection
review_sizes = len(list(review_data['asin']))

for i in range(review_sizes):
    # list of relevant_data = [awesomeness, [summ score],[text score], [votes], [time], [verified], [images]]
    summary_text = lemmatize(review_data['summaries'][i])
    
    new_addition = get_sentiment(summary_text)
    c = temp_dict[review_data['asin'][i]]
    final_summary = [c[0], c[1] + new_addition, c[2], c[3],c[4],c[5],c[6],c[7]]
    temp_dict[review_data['asin'][i]] =  final_summary
    



        
   
    
def lemmatize(mystr):
    return mystr

def get_sentiment(my_str):
    
    return 0



