#import csv
#import json
import nltk
#from nltk.sentiment.vader import SentimentIntensityAnalyzer
nltk.download('vader_lexicon')
nltk.download('stopwords')
#from nltk.stem import WordNetLemmatizer
nltk.download('wordnet')
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
import warnings
import pandas as pd
#from nltk.corpus import stopwords
#from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV


# Collecting all training files and cleaning up None values
review_path = "devided_dataset_v2/CDs_and_Vinyl/train/review_training.json"
product_path = "devided_dataset_v2/CDs_and_Vinyl/train/product_training.json"
product_data = pd.read_json(product_path)
review_data = pd.read_json(review_path)
review_data["summary"] = review_data["summary"].fillna("negative")
review_data["reviewText"] = review_data["reviewText"].fillna("negative")
review_data['vote'] = review_data['vote'].apply(lambda x: 0 if x == None else int(x.replace(',','')))
review_data['image'] = review_data['image'].apply(lambda x: False if x == None else True)

# Collecting all testing files and cleaning up None values
test_review_path = "devided_dataset_v2/CDs_and_Vinyl/test1/review_test.json"
test_product_path = "devided_dataset_v2/CDs_and_Vinyl/test1/product_test.json"
test_product_data = pd.read_json(test_product_path)
test_review_data = pd.read_json(test_review_path)
test_review_data["summary"] = test_review_data["summary"].fillna("negative")
test_review_data["reviewText"] = test_review_data["reviewText"].fillna("negative")
test_review_data['vote'] = test_review_data['vote'].apply(lambda x: 0 if x == None else int(x.replace(',','')))
test_review_data['image'] = test_review_data['image'].apply(lambda x: False if x == None else True)

# Initializing vectorizer and corpus for the TF-IDF score calculations
id_vector_awesome = product_data[product_data["awesomeness"] == 1]['asin']
df_awesome_corpus = review_data.loc[review_data['asin'].isin(id_vector_awesome), ['asin', 'summary']].reset_index(drop=True)
awesome_corpus = list(df_awesome_corpus.loc[:, 'summary']) # awesome corpus

id_vector_notawesome = product_data[product_data["awesomeness"] == 0]['asin']
df_notawesome_corpus = review_data.loc[review_data['asin'].isin(id_vector_notawesome), ['asin', 'summary']].reset_index(drop=True)
notawesome_corpus = list(df_notawesome_corpus.loc[:, 'summary']) # not awesome corpus

vectorizer_awesome = TfidfVectorizer(stop_words='english') # Awesome CountVectorizer object
vectorizer_notawesome = TfidfVectorizer(stop_words='english') # Not Awesome CountVectorizer object

train_tfidf_awesome = vectorizer_awesome.fit_transform(awesome_corpus) # Fit the vectorizer to the corpus for awesome
train_dtfidf_notawesome = vectorizer_notawesome.fit_transform(notawesome_corpus) # Fit the vectorizer to the corpus for not awesome

vocabulary_awesome = vectorizer_awesome.get_feature_names() # Awesome vocabulary
vocabulary_notawesome = vectorizer_notawesome.get_feature_names() # Not awesome vocabulary

# slice the data to retain only the first 1000 rows
#product_data = product_data.iloc[:1000, :]

warnings.filterwarnings("ignore")


def get_TFIDF(text):
    
    # Transform the testing reviews into TF-IDF scores using the same vectorizer
    test_tfidf_awesome = vectorizer_awesome.transform(text)
    test_tfidf_not_awesome = vectorizer_notawesome.transform(text)

    # Transforming document-term matrix to a dataframe
    test_tfidf_awesome_df = pd.DataFrame(test_tfidf_awesome.toarray(), columns=vocabulary_awesome)
    test_tfidf_not_awesome_df = pd.DataFrame(test_tfidf_not_awesome.toarray(), columns=vocabulary_notawesome)

    # Sum the columns together and create a new column with the results
    aggregate_tfidf_awesome = test_tfidf_awesome_df.sum(axis=1)
    aggregate_tfidf_not_awesome = test_tfidf_not_awesome_df.sum(axis=1)

    return [aggregate_tfidf_awesome, aggregate_tfidf_not_awesome]


def is_positive(df, text1, text2):
    return df[text1].mean() > df[text2].mean()


def is_negative(df, text1, text2):
    return df[text1].mean() < df[text2].mean()


def get_vote_score(df):
    
    pos = 0
    neg = 0 
    pos_list1, neg_list1  = list(df['aw_rt']), list(df['naw_rt'])
    pos_list2, neg_list2  = list(df['aw_s']), list(df['naw_s'])
    vote_list = list(df['vote'])
    i = 0
    
    while i < len(pos_list1):
        
        temp_pos = pos
        temp_neg = neg
        
        if pos_list1[i] > neg_list1[i]:
            pos += vote_list[i]
        elif pos_list1[i] < neg_list1[i]:
            neg += vote_list[i]
            
        if (pos_list2[i] > neg_list2[i]) and temp_pos == pos:
            pos += vote_list[i]
        elif pos_list2[i] < neg_list2[i] and temp_neg == neg:
            neg += vote_list[i]
        
        i += 1
    

    return (pos + 1)/(neg + 1)


def calculate_time_score(df):
    
    """function the gives a value between 0 and 1 for a review depending
    on the relative time it was sent
    """
    
    oldest_date = df['unixReviewTime'].min()
    newest_date = df['unixReviewTime'].max()
    
    if oldest_date == newest_date:
        return 1  # if there's only one review, give it a neutral score
    
    pos_scores = []
    neg_scores = []
    
    time_range = newest_date - oldest_date
    
    for index, row in df.iterrows():
        
        date = row['unixReviewTime']
        score = (date - oldest_date) / time_range
        
        if row['aw_rt'] > row['naw_rt']:
            pos_scores.append(score)
        else:
            neg_scores.append(score)
            
    if not pos_scores:
        pos_scores.append(0.5)
    if not neg_scores:
        neg_scores.append(0.5)
        
        
    return ((sum(pos_scores)/len(pos_scores)) + 1) / ((sum(neg_scores)/len(neg_scores))+1)



def image_review_count(df):
    """ function to count number of positive reviews with images and 
    negative reviews with images"""
    
    df_new = df[df["image"]==True]
    pos1 = is_positive(df_new, 'aw_rt', 'naw_rt')
    neg1 = is_negative(df_new, 'aw_rt', 'naw_rt')
     
    
    pos2 = is_positive(df_new, 'aw_s', 'naw_s')
    neg2 = is_negative(df_new, 'aw_s', 'naw_s')
    
    
    return ((pos1 + pos2)+1) / ((neg1 + neg2) + 1)

def num_verified(df):

    """function to count number of positive verified and negative verified"""

    df_new = df[df["verified"]==True]
    pos1 = is_positive(df_new, 'aw_rt', 'naw_rt')
    neg1 = is_negative(df_new, 'aw_rt', 'naw_rt')
     
    
    pos2 = is_positive(df_new, 'aw_s', 'naw_s')
    neg2 = is_negative(df_new, 'aw_s', 'naw_s')
    
    
    return ((pos1 + pos2) + 1) / ( (neg1 + neg2) + 1)

    
feature_vector = pd.DataFrame({"aw_rt":[], "naw_rt":[], "aw_s":[], "naw_s":[],"vote_score":[], "image_score":[], "verified":[], "time_score":[]})
iDs = list(product_data['asin'])

def make_feature_vector(iDs, feature_vector, review_data):
    
    """loop through the products and construct the row with the feature vector 
    values"""
    
    k = 0
    for i in iDs:
        if (k % 1000 == 0):
            print(k)
        k += 1
        current_data = review_data[review_data["asin"] == i]
        aw_rt, naw_rt  = get_TFIDF(list(current_data['reviewText']))
        aw_s, naw_s  = get_TFIDF(list(current_data['summary']))
        
        current_data['aw_rt'] = list(aw_rt)
        current_data['naw_rt'] = list(naw_rt)
        current_data['aw_s'] = list(aw_s)
        current_data['naw_s'] = list(naw_s)
        
        vote_score = get_vote_score(current_data)
        image_score = image_review_count(current_data)
        verified = num_verified(current_data)
        
        aw_rt1 = current_data['aw_rt'].mean()
        naw_rt1 = current_data['naw_rt'].mean()
        aw_s1 = current_data['aw_s'].mean()
        naw_s1 = current_data['naw_s'].mean()
        time = calculate_time_score(current_data)
        
        feature_vector.loc[len(feature_vector)] = [aw_rt1, naw_rt1, aw_s1, naw_s1, vote_score, image_score, verified, time]
    return feature_vector

train_feature_vector = make_feature_vector(iDs, feature_vector, review_data)
train_feature_vector["awesomeness"] = list(product_data["awesomeness"])



    

#NOW TRAINING
X = train_feature_vector.iloc[:, :-1]
y = train_feature_vector.iloc[:, -1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
clf = GradientBoostingClassifier()
clf.fit(X_train, y_train)
score = clf.score(X_test, y_test)


# Define the parameter grid
param_grid = {
    'learning_rate': [0.1, 0.01],
    'n_estimators': [100, 200],
    'max_depth': [3, 5],
    'min_samples_split': [2, 4] }

# Create the model
clf = GradientBoostingClassifier()

# Performing Hyperparameter Optimization - perform grid search cross-validation
grid_search = GridSearchCV(clf, param_grid, cv=5)
grid_search.fit(X_train, y_train)

# Print the best hyperparameters
print("Best hyperparameters:", grid_search.best_params_)

# Train the model with the best hyperparameters
clf = GradientBoostingClassifier(**grid_search.best_params_)
clf.fit(X_train, y_train)

# Calculating Score
score = clf.score(X_test, y_test)

train_feature_vector.to_csv("feature_vectorv2.csv")

'''
#Now for testing
iDs = list(test_product_data['asin'])
feature_vector_2 = pd.DataFrame({"num_pos":[], "num_neg":[], "vote_score":[], "pos_image_count":[], "neg_image_count":[], "pos_verified_count":[], "neg_verified_count":[], "pos_time_score":[], "neg_time_score":[]})
test_feature_vector = make_feature_vector(iDs, feature_vector_2, test_review_data)
predicted_class = clf.predict(test_feature_vector)

final_json = test_product_data
final_json["awesomeness"] = predicted_class
final_json.to_json("predictions.json")'''










