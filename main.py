import pandas as pd
import csv
import json
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
nltk.download('vader_lexicon')
nltk.download('stopwords')
from nltk.stem import WordNetLemmatizer
nltk.download('wordnet')
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
import warnings

warnings.filterwarnings("ignore")



# Collecting all files and cleaning up None values
review_path = "devided_dataset_v2/CDs_and_Vinyl/train/review_training.json"
product_path = "devided_dataset_v2/CDs_and_Vinyl/train/product_training.json"
product_data = pd.read_json(product_path)
review_data = pd.read_json(review_path)
review_data["summary"] = review_data["summary"].fillna("negative")
review_data["reviewText"] = review_data["reviewText"].fillna("negative")
review_data['vote'] = review_data['vote'].apply(lambda x: 0 if x == None else int(x.replace(',','')))
review_data['image'] = review_data['image'].apply(lambda x: False if x == None else True)


test_review_path = "devided_dataset_v2/CDs_and_Vinyl/test1/review_test.json"
test_product_path = "devided_dataset_v2/CDs_and_Vinyl/test1/product_test.json"
test_product_data = pd.read_json(test_product_path)
test_review_data = pd.read_json(test_review_path)
test_review_data["summary"] = test_review_data["summary"].fillna("negative")
test_review_data["reviewText"] = test_review_data["reviewText"].fillna("negative")
test_review_data['vote'] = test_review_data['vote'].apply(lambda x: 0 if x == None else int(x.replace(',','')))
test_review_data['image'] = test_review_data['image'].apply(lambda x: False if x == None else True)


# slice the data to retain only the first 1000 rows
product_data = product_data.iloc[:10000, :]


'''
 Our feature vector  is as follows 

#1 num_pos ->  ratio of positive reviews and summaries (based on sentiments)
#2 num_neg -> ratio of negative reviews and summaries
#3 vote_score -> weighting positive votes vs negatuve votes
#4 pos_image_count -> counting which positive reviews have images
#5 neg_image_count -> counting which negative reviews have images
#6 pos_verified_count -> counting which positive reviews are verified
#7 neg_verified_count -> counting which negative reviews are verified
#8 pos_time_score
#9 neg_time_score

'''
awesome_corpus = ""
non_awesome_corpus = ""
def get_TFIDF(text, choices):
    
    result = process.extractBests(text, choices, score_cutoff=0, limit=None, scorer = fuzz.token_sort_ratio) 
    return [result[0][1], result[1][1]]
    



global analyzer
analyzer = SentimentIntensityAnalyzer()
def get_sentiment(text):
    ''' function to get the sentiment of a text'''
    global analyzer
    scores = analyzer.polarity_scores(text)
    return scores['compound']
    
def add_sentiment_col(df): 
    '''adds a columnn to dataframe; indicating positive or negative sentiment'''
    
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


feature_vector = pd.DataFrame({"num_pos":[], "num_neg":[], "vote_score":[], "pos_image_count":[], "neg_image_count":[], "pos_verified_count":[], "neg_verified_count":[], "pos_time_score":[], "neg_time_score":[]})
iDs = list(product_data['asin'])

# create a WordNetLemmatizer object
lemmatizer = WordNetLemmatizer()

# remove stopwords from the list of words
stopwords_list = stopwords.words('english')
filtered_words = [word for word in words if word not in stopwords_list]

# define a function to lemmatize and remove stopwords from a sentence
def lemmatize_sentence(sentence, filtered_words):
    # tokenize the sentence into words
    words = nltk.word_tokenize(sentence.lower())
    
    # lemmatize each word in the filtered list of words
    lemmatized_words = [lemmatizer.lemmatize(word) for word in filtered_words]
    
    # join the lemmatized words back into a sentence
    return ' '.join(lemmatized_words)

def get_all_words(df, col_name, id_list):
    #filter data
    df = df[df['asin'].isin(id_list)]
    #concatenate
    concatenated_text = df[col_name].str.cat(sep=' ')

    return lemmatize_sentence(concatenated_text)

def make_feature_vector(iDs, feature_vector, review_data):
    
    """loop through the products and construct the row with the feature vector 
    values"""
    k = 0
    for i in iDs:
        if (k % 1000 == 0):
            print(k)
        k += 1
        current_data = review_data[review_data["asin"] == i]
        text_sentiments = add_sentiment_col(current_data)
        num_pos, num_neg = get_num_pos_neg(text_sentiments)
        pos_ratio = (num_pos)/(num_neg + num_pos)
        neg_ratio = (num_neg)/(num_pos + num_neg)
        vote_score = get_vote_score(text_sentiments)
        pos_image_count, neg_image_count = image_review_count(text_sentiments)
        pos_verified_count, neg_verified_count = num_verified(text_sentiments)
        pos_time_score, neg_time_score  = calculate_time_score(text_sentiments)
        feature_vector.loc[len(feature_vector)] = [pos_ratio, neg_ratio, vote_score, pos_image_count, neg_image_count, pos_verified_count, neg_verified_count, pos_time_score, neg_time_score]
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

#Now for testing
iDs = list(test_product_data['asin'])
feature_vector_2 = pd.DataFrame({"num_pos":[], "num_neg":[], "vote_score":[], "pos_image_count":[], "neg_image_count":[], "pos_verified_count":[], "neg_verified_count":[], "pos_time_score":[], "neg_time_score":[]})
test_feature_vector = make_feature_vector(iDs, feature_vector_2, test_review_data)
predicted_class = clf.predict(test_feature_vector)

final_json = test_product_data
final_json["awesomeness"] = predicted_class
final_json.to_json("predictions.json")










