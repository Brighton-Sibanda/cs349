import pandas as pd
import csv
import json
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from datetime import datetime
nltk.download('vader_lexicon')
nltk.download('stopwords')


review_path = "devided_dataset_v2/CDs_and_Vinyl/train/review_training.json"
product_path = "devided_dataset_v2/CDs_and_Vinyl/train/product_training.json"
product_data = pd.read_json(product_path)
review_data = pd.read_json(review_path)


'''
#Training and coming up with feature vector
avg positive summary 
#2 avg negative summary
#3 avg positive review text
#4 avg negative review text
#5 avg vote credibility score
#6 image credibility score
#7 verified credibility score
#8 time score

'''
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
    positive = []

    for i in df['summary']:
        sent = get_sentiment(i)
        if sent == 'positive':
            positive.append('True')
        else:
            positive.append('False')
        
    df['positive'] = positive
    return df

# now to calculate averages with weights accounted for

def get_num_votes(df):
    """ 
    input: pandas dataframe with all reviews for one product
    output: two-element tuple with first element being the 
            number of positive reviews and the second element 
            being the number of negative reviews
    """
    pos = len(df[df["positive"]])
    neg = len(df[df["positive"] == False])
    return (pos, neg)

def calculate_time_score(reviews):
    """function the gives a value between 0 and 1 for a review depending
    on the relative time it was sent
    """
    oldest_date = min(review['date'] for review in reviews)
    newest_date = max(review['date'] for review in reviews)
    
    if oldest_date == newest_date:
        return 0.5  # if there's only one review, give it a neutral score
    
    time_scores = []
    for review in reviews:
        date = review['date']
        time_diff = (date - oldest_date).days / (newest_date - oldest_date).days
        time_score = 1 - time_diff
        time_scores.append(time_score)
    
    return sum(time_scores) / len(time_scores) # add something



def replace_none_with_zero(df):
    """ function to substitute 'None' for False and True otherwise for 
     the images column
    """
    df['image'] = df['image'].apply(lambda x: False if x == 'None' else True)
    return df


def image_review_count(df):
    """ function to count number of positive reviews with images and 
    negative reviews with images"""
    
    df = df[df["image"]]
    pos = len(df[df["positive"]])
    neg = len(df[df["positive"]==False])
    return (pos, neg)

def num_verified(df):

    """function to count number of positive verified and negative verified"""

    df = df[df["verified"]]
    pos = len(df[df["positive"]])
    neg = len(df[df["positive"]==False])
    return (pos, neg)

"""loop through the products and construct the row with the feature vector 
values"""


