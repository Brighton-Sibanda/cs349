import pandas as pd
import csv
import json
from textblob import TextBlob
from textblob import Word
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
nltk.download('vader_lexicon')
nltk.download('stopwords')


review_path = "devided_dataset_v2/CDs_and_Vinyl/train/review_training.json"
product_path = "devided_dataset_v2/CDs_and_Vinyl/train/product_training.json"
product_data = pd.read_json(product_path)
review_data = pd.read_json(review_path)


'''
#Training and coming up with feature vector
feature_vector = {}  # key = product ; values = array of final feature values
temp_dict = {} # temp dict to compile current feature info for a product accross features

productIDs = product_data
length = len(list(productIDs['asin']))
for i in range(length):
    feature_vector[productIDs['asin'][i]] = [productIDs['awesomeness'][i]]
    temp_dict[productIDs['asin'][i]] = [[], [], [],[],[],[],[]]
    
#order array indices according to order numbered here; for reature vector

#1 avg positive summary 
#2 avg negative summary
#3 avg positive review text
#4 avg negative review text
#5 avg vote credibility score
#6 image credibility score
#7 verified credibility score
#8 time score

#data collection functions
review_sizes = len(list(review_data['asin']))
def lemmatize(mystr):
    stop_words = set(nltk.corpus.stopwords.words('english'))
    words = TextBlob(mystr).words
    words = [word.lemmatize() for word in words if word.lower() not in stop_words]
    lemmatized_text = " ".join(words)
    return lemmatized_text

# now collecting relevant information and making conversions for all reviews
for i in range(review_sizes):
    # list of relevant_data = [[summ scores],[text scores], [votes], [times], [verifieds], [images]]
    
    # for summary
    
    
    #for reviewtext
    review_text = lemmatize(review_data['reviewText'][i])
    new_addition = get_sentiment(review_text)
    c = temp_dict[review_data['asin'][i]]
    final_review = [c[0], c[1]+ [new_addition], c[2],c[3],c[5],c[5],c[6]]
    temp_dict[review_data['asin'][i]] =  final_review
    
    # for vote score
    vote = review_data['vote'][i]
    c = temp_dict[review_data['asin'][i]]
    final_votes = [c[0], c[1], c[2]+ [vote],c[3],c[4],c[5],c[6]]
    temp_dict[review_data['asin'][i]] =  final_votes

    # for time
    time = review_data['ReviewTime'][i]
    c = temp_dict[review_data['asin'][i]]
    final_time = [c[0], c[1], c[2], c[3]+[time],c[4],c[5],c[6]]
    temp_dict[review_data['asin'][i]] =  final_time

    # for image
    image = review_data['image'][i]
    if image == 'none':
        image = 0
    else:
        image = 1
    c = temp_dict[review_data['asin'][i]]
    final_image = [c[0], c[1], c[2], c[3], c[4]+[image],c[5],c[6]]
    temp_dict[review_data['asin'][i]] =  final_image

    # for verified
    verified = review_data['verified'][i]
    c = temp_dict[review_data['asin'][i]]
    final_verified = [c[0], c[1], c[2], c[3], c[4]+[verified],c[5],c[6]]
    temp_dict[review_data['asin'][i]] =  final_verified


    # now to calculate averages with weights accounted for'''
def get_sentiment(text):
    analyzer = SentimentIntensityAnalyzer()
    scores = analyzer.polarity_scores(text)
    # Get the sentiment label based on the polarity scores
    if scores['compound'] > 0:
        sentiment_label = 'Positive'
    elif scores['compound'] < 0:
        sentiment_label = 'Negative'
    else:
        sentiment_label = 'Neutral'
    return sentiment_label
    
def get_text(df): 
    num_positives = 0
    num_negatives = 0
    for i in df['summary']:
        sent = get_sentiment(i)
        if sent == 'positive':
            num_positives += 1
        elif sent == 'negative':
            num_negatives += 1
    return [num_positives, num_negatives]


    



