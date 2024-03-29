from thefuzz import fuzz, process
import csv
import json
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
nltk.download('vader_lexicon')
nltk.download('stopwords')
from nltk.stem import WordNetLemmatizer
nltk.download('wordnet')
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
import warnings
import pandas as pd
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

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
product_data = product_data.iloc[:100, :]

warnings.filterwarnings("ignore")

lemmatizer = WordNetLemmatizer()

# define a function to lemmatize and remove stopwords from a sentence
def lemmatize_sentence(sentence):
    # tokenize the sentence into words
    words = nltk.word_tokenize(sentence.lower())
    # remove stopwords from the list of words
    stopwords_list = stopwords.words('english')
    filtered_words = [word for word in words if word not in stopwords_list]
    # lemmatize each word in the filtered list of words
    lemmatized_words = [lemmatizer.lemmatize(word) for word in filtered_words]
    
    # join the lemmatized words back into a sentence
    return ' '.join(lemmatized_words)

def get_all_words(df, col_name, id_list, code):
    #filter data
    if code == "corpus":
        df = df[df['asin'].isin(id_list)]
    
    #concatenate
    concatenated_text = df[col_name].str.cat(sep=' ')

    return lemmatize_sentence(concatenated_text )

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

    return aggregate_tfidf_awesome, aggregate_tfidf_not_awesome

def build_corpus(product_data, review_data, text):
    
    awesome = list(product_data[product_data["awesomeness"] == 1]['asin'])
    non_awesome = list(product_data[product_data["awesomeness"] == 0]['asin'])
    
   # awesome_df = review_data[review_data['asin'] in awesome]
    # non_awesome_df = review_data[review_data['asin'] in awesome]
    
    return [get_all_words(review_data, text, awesome,"corpus"), get_all_words(review_data, text, non_awesome, "corpus")]


def is_positive(text):
    return get_TFIDF(text, awesome_reviews_corpus) > get_TFIDF(text, non_awesome_reviews_corpus)
def is_negative(text):
    return get_TFIDF(text, awesome_reviews_corpus) < get_TFIDF(text, non_awesome_reviews_corpus)


def get_vote_score(df):

    pos = df[df['reviewText'].apply(is_positive)]
    pos = pos['vote'].sum()
    neg = df[df['reviewText'].apply(is_negative)]
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
    pos1 = get_TFIDF(get_all_words(df_new, 'reviewText', list(df_new['asin']), "data"), awesome_reviews_corpus)
    neg1 = get_TFIDF(get_all_words(df_new, 'reviewText', list(df_new['asin']), "data"), non_awesome_reviews_corpus)
    
    pos2 = get_TFIDF(get_all_words(df_new, 'summary', list(df_new['asin']), "data"), awesome_reviews_corpus)
    neg2 = get_TFIDF(get_all_words(df_new, 'summary', list(df_new['asin']), "data"), non_awesome_reviews_corpus)
    
    return (pos1 + pos2) - (neg1 + neg2)

def num_verified(df):

    """function to count number of positive verified and negative verified"""

    df_new = df[df["verified"]==True]
    pos1 = get_TFIDF(get_all_words(df_new, 'reviewText', list(df_new['asin']), "data"), awesome_reviews_corpus)
    neg1 = get_TFIDF(get_all_words(df_new, 'reviewText', list(df_new['asin']), "data"), non_awesome_reviews_corpus)
    
    pos2 = get_TFIDF(get_all_words(df_new, 'summary', list(df_new['asin']), "data"), awesome_reviews_corpus)
    neg2 = get_TFIDF(get_all_words(df_new, 'summary', list(df_new['asin']), "data"), non_awesome_reviews_corpus)
    
    return (pos1 + pos2) - (neg1 + neg2)



awesome_reviews_corpus, non_awesome_reviews_corpus = build_corpus(product_data, review_data, 'reviewText')
awesome_summaries_corpus, non_awesome_summaries_corpus = build_corpus(product_data, review_data, 'summary')
    
feature_vector = pd.DataFrame({"aw_rt":[], "naw_rt":[], "aw_s":[], "naw_s":[],"vote_score":[], "image_score":[], "verified":[]})
iDs = list(product_data['asin'])

def make_feature_vector(iDs, feature_vector, review_data):
    
    """loop through the products and construct the row with the feature vector 
    values"""
    
    k = 0
    for i in iDs:
        if (k % 1 == 0):
            print(k)
        k += 1
        current_data = review_data[review_data["asin"] == i]
        aw_rt = get_TFIDF(get_all_words(current_data, 'reviewText', list(i), "data"), awesome_reviews_corpus)
        naw_rt = get_TFIDF(get_all_words(current_data, 'reviewText', list(i), "data"), non_awesome_reviews_corpus)
        
        aw_s = get_TFIDF(get_all_words(current_data, 'summary', list(i), "data"), awesome_reviews_corpus)
        naw_s = get_TFIDF(get_all_words(current_data, 'summary', list(i), "data"), non_awesome_reviews_corpus)
        
        vote_score = get_vote_score(current_data)
        image_score = image_review_count(current_data)
        verified = num_verified(current_data)
        
        
        feature_vector.loc[len(feature_vector)] = [aw_rt, naw_rt, aw_s, naw_s, vote_score, image_score, verified]
    return feature_vector

train_feature_vector = make_feature_vector(iDs, feature_vector, review_data)
train_feature_vector["awesomeness"] = list(product_data["awesomeness"])



    