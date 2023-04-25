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
    else:
        sentiment_label = 'Negative'
    return sentiment_label 
    
def avg_pos_neg_sent(df, col): 
    positive = []

    for i in df[col]:
        sent = get_sentiment(i)
        if sent == 'Positive':
            positive.append('True')
        else:
            positive.append('False')
        
    df['positive'] = positive
    return df

# now to calculate averages with weights accounted for

def get_num_pos_neg(df):
    """ 
    input: pandas dataframe with all reviews for one product
    output: two-element tuple with first element being the 
            number of positive reviews and the second element 
            being the number of negative reviews
    """
    pos = len(df[df["positive"]])
    neg = len(df[df["positive"] == False])
    return (pos, neg)

def get_vote_score(df):


    pos = df[df["positive"]]
    pos = pos['vote'].sum()
    neg = df[df["positive"]==False]
    neg = neg['vote'].sum()

    return (pos + 1)/(neg + 1)
    

def calculate_time_score(reviews):
    """function the gives a value between 0 and 1 for a review depending
    on the relative time it was sent
    """
    oldest_date = reviews.iloc[:, 2].min()
    newest_date = reviews.iloc[:, 2].max()
    
    if oldest_date == newest_date:
        return 0.5  # if there's only one review, give it a neutral score
    
    time_scores = []
    for i in range(len(reviews)):
        date = reviews.iloc[:, 2]
        time_diff = (date - oldest_date) / (newest_date - oldest_date)
        time_score = 1 - time_diff
        time_scores.append(time_score)
    
    return sum(time_scores) / len(time_scores) 

def replace_none_with_zero(df):
    """ function to substitute 'None' for False and True otherwise for 
     the images column
    """
    df['image'] = df['image'].apply(lambda x: False if x == 'None' else True)
    return df


def image_review_count(df):
    """ function to count number of positive reviews with images and 
    negative reviews with images"""
    
    df_new = df[df["image"]]
    pos = len(df[df_new["positive"]])
    neg = len(df[df_new["positive"]==False])
    return (pos, neg)

def num_verified(df):

    """function to count number of positive verified and negative verified"""

    df_new = df[df["verified"]]
    pos = len(df[df_new["positive"]])
    neg = len(df[df_new["positive"]==False])
    return (pos, neg)

"""loop through the products and construct the row with the feature vector 
values"""