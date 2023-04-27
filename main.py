import pandas as pd
import csv
import json
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from datetime import datetime
nltk.download('vader_lexicon')
nltk.download('stopwords')

""
review_path = "devided_dataset_v2/CDs_and_Vinyl/train/review_training.json"
product_path = "devided_dataset_v2/CDs_and_Vinyl/train/product_training.json"
product_data = pd.read_json(product_path)
review_data = pd.read_json(review_path)
review_data["summary"] = review_data["summary"].fillna("negative")
review_data["reviewText"] = review_data["reviewText"].fillna("negative")
review_data['vote'] = review_data['vote'].apply(lambda x: 0 if x == None else int(x.replace(',','')))
review_data['image'] = review_data['image'].apply(lambda x: False if x == None else True)


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
    return scores['compound']
    
def add_sentiment_col(df): 
    positive = []

    for index, row in df.iterrows():
        summary = row["summary"]
        text = row["reviewText"]
        summary_score = get_sentiment(summary)
        text_score = get_sentiment(text)
        if summary_score == 0:
            sentiment_score = 2*text_score
        elif text_score == 0:
            sentiment_score = 2 * summary_score
        else:
            sentiment_score = summary_score + text_score

        if sentiment_score>0.6:
            positive.append(True)
        else:
            positive.append(False)
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
    pos = len(df[df["positive"] == True])
    neg = len(df[df["positive"] == False])
    return (pos, neg)

def get_vote_score(df):


    pos = df[df["positive"] == True]
    pos = pos['vote'].sum()
    neg = df[df["positive"]==False]
    neg = neg['vote'].sum()

    return (pos + 1)/(neg + 1)

def calculate_time_score(df):
    """function the gives a value between 0 and 1 for a review depending
    on the relative time it was sent
    """
    oldest_date = df['unixReviewTime'].min()
    newest_date = df['unixReviewTime'].max()
    
    if oldest_date == newest_date:
        return (0.5, 0.5)  # if there's only one review, give it a neutral score
    
    pos_scores = []
    neg_scores = []
    time_range = newest_date - oldest_date
    for index, row in df.iterrows():
        date = row['unixReviewTime']
        score = (date - oldest_date) / time_range
        if row["positive"]:
            pos_scores.append(score)
        else:
            neg_scores.append(score)
    if not pos_scores:
        pos_scores.append(0.5)
    if not neg_scores:
        neg_scores.append(0.5)
    return (sum(pos_scores)/len(pos_scores), sum(neg_scores)/len(neg_scores))

def image_review_count(df):
    """ function to count number of positive reviews with images and 
    negative reviews with images"""
    
    df_new = df[df["image"]]
    pos = len(df_new[df_new["positive"] == True])
    neg = len(df_new[df_new["positive"]==False])
    return (pos, neg)

def num_verified(df):

    """function to count number of positive verified and negative verified"""

    df_new = df[df["verified"]==True]
    pos = len(df_new[df_new["positive"]==True])
    neg = len(df_new[df_new["positive"]==False])
    return (pos, neg)

"""loop through the products and construct the row with the feature vector 
values"""

feature_vector = pd.DataFrame({"asin":[],"num_pos":[], "num_neg":[], "vote_score":[], "pos_image_count":[], "neg_image_count":[], "pos_verified_count":[], "neg_verified_count":[], "pos_time_score":[], "neg_time_score":[]})
iDs = list(product_data['asin'])

for i in iDs:
    current_data = review_data[review_data["asin"] == i]
    text_sentiments = add_sentiment_col(current_data)
    num_pos, num_neg = get_num_pos_neg(text_sentiments)
    pos_ratio = (num_pos)/(num_neg + num_pos)
    neg_ratio = (num_neg)/(num_pos + num_neg)
    vote_score = get_vote_score(text_sentiments)
    pos_image_count, neg_image_count = image_review_count(text_sentiments)
    pos_verified_count, neg_verified_count = num_verified(text_sentiments)
    pos_time_score, neg_time_score  = calculate_time_score(text_sentiments)

    feature_vector.loc[len(feature_vector)] = [i,num_pos, num_neg, vote_score, pos_image_count, neg_image_count, pos_verified_count, neg_verified_count, pos_time_score, neg_time_score]
feature_vector["awesomeness"] = list(product_data["awesomeness"])



