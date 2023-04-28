import pandas as pd
import csv
import json
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
nltk.download('vader_lexicon')
nltk.download('stopwords')
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

def get_sentiment(text):
    ''' function to get the sentiment of a text'''
    analyzer = SentimentIntensityAnalyzer()
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


# CODE FOR EVALUATION; COMMENTED OUT TO SAVE TIME
''' #METRICS FOR MODELS THAT WE TRAINED: 

from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import cross_val_score


feature_vector = train_feature_vector

# Separating features and target variables in training data
train_ind = feature_vector.iloc[:, 2:-1]
train_dep = feature_vector.iloc[:, -1]

# Creating models
nb_model = MultinomialNB()
knn_model = KNeighborsClassifier()
dt_model = DecisionTreeClassifier()
svm_model = SVC()
lr_model = LogisticRegression()
rf_model = RandomForestClassifier()
gb_model = GradientBoostingClassifier()

# Training the models
nb_model.fit(train_ind, train_dep)
knn_model.fit(train_ind, train_dep)
dt_model.fit(train_ind, train_dep)
svm_model.fit(train_ind, train_dep)
lr_model.fit(train_ind, train_dep)
rf_model.fit(train_ind, train_dep)
gb_model.fit(train_ind, train_dep)

# Predicting the classification of training data for each model
nb_y_pred = nb_model.predict(train_ind)
knn_y_pred = knn_model.predict(train_ind)
dt_y_pred = dt_model.predict(train_ind)
svm_y_pred = svm_model.predict(train_ind)
lr_y_pred = lr_model.predict(train_ind)
rf_y_pred = rf_model.predict(train_ind)
gb_y_pred = gb_model.predict(train_ind)

# Calculating precision, recall, and F1 score of each model on training data
nb_precision = precision_score(train_dep, nb_y_pred, average='macro')
nb_recall = recall_score(train_dep, nb_y_pred, average='macro')
nb_f1 = f1_score(train_dep, nb_y_pred, average='macro')
print("Multinomial Naive Bayes Precision: {:.2f}".format(nb_precision))
print("Multinomial Naive Bayes Recall: {:.2f}".format(nb_recall))
print("Multinomial Naive Bayes F1 Score: {:.2f}".format(nb_f1))

knn_precision = precision_score(train_dep, knn_y_pred, average='macro')
knn_recall = recall_score(train_dep, knn_y_pred, average='macro')
knn_f1 = f1_score(train_dep, knn_y_pred, average='macro')
print("K-Nearest Neighbor Precision: {:.2f}".format(knn_precision))
print("K-Nearest Neighbor Recall: {:.2f}".format(knn_recall))
print("K-Nearest Neighbor F1 Score: {:.2f}".format(knn_f1))


dt_precision = precision_score(train_dep, dt_y_pred, average='macro')
dt_recall = recall_score(train_dep, dt_y_pred, average='macro')
dt_f1 = f1_score(train_dep, dt_y_pred, average='macro')
print("Decision Tree Precision: {:.2f}".format(dt_precision))
print("Decision Tree Recall: {:.2f}".format(dt_recall))
print("Decision Tree F1 Score: {:.2f}".format(dt_f1))

svm_precision = precision_score(train_dep, svm_y_pred, average='macro')
svm_recall = recall_score(train_dep, svm_y_pred, average='macro')
svm_f1 = f1_score(train_dep, svm_y_pred, average='macro')
print("Support Vector Machine Precision: {:.2f}".format(svm_precision))
print("Support Vector Machine Recall: {:.2f}".format(svm_recall))
print("Support Vector Machine F1 Score: {:.2f}".format(svm_f1))

lr_precision = precision_score(train_dep, lr_y_pred, average='macro')
lr_recall = recall_score(train_dep, lr_y_pred, average='macro')
lr_f1 = f1_score(train_dep, lr_y_pred, average='macro')
print("Logistic Regression Precision: {:.2f}".format(lr_precision))
print("Logistic Regression Recall: {:.2f}".format(lr_recall))
print("Logistic Regression F1 Score: {:.2f}".format(lr_f1))

rf_precision = precision_score(train_dep, rf_y_pred, average='macro')
rf_recall = recall_score(train_dep, rf_y_pred, average='macro')
rf_f1 = f1_score(train_dep, rf_y_pred, average='macro')
print("Random Forest Precision: {:.2f}".format(rf_precision))
print("Random Forest Recall: {:.2f}".format(rf_recall))
print("Random Forest F1 Score: {:.2f}".format(rf_f1))

gb_precision = precision_score(train_dep, gb_y_pred, average='macro')
gb_recall = recall_score(train_dep, gb_y_pred, average='macro')
gb_f1 = f1_score(train_dep, gb_y_pred, average='macro')
print("Gradient Boosting Precision: {:.2f}".format(gb_precision))
print("Gradient Boosting Recall: {:.2f}".format(gb_recall))
print("Gradient Boosting F1 Score: {:.2f}".format(gb_f1))

knn_scores = cross_val_score(knn_model, train_ind, train_dep, cv=10, scoring='f1_macro')
print("k nearest neighbor f1 Score (10-fold cross-validation): {:.2f}".format(knn_scores.mean()))

svm_scores = cross_val_score(svm_model, train_ind, train_dep, cv=10, scoring='f1_macro')
print("Support Vector Machine F1 Score (10-fold cross-validation): {:.2f}".format(svm_scores.mean()))

lr_scores = cross_val_score(lr_model, train_ind, train_dep, cv=10, scoring='f1_macro')
print("Logistic Regression F1 Score (10-fold cross-validation): {:.2f}".format(lr_scores.mean()))

gb_scores = cross_val_score(gb_model, train_ind, train_dep, cv=10, scoring='f1_macro')
print("Gradient Boosting F1 Score (10-fold cross-validation): {:.2f}".format(gb_scores.mean()))

nb_scores = cross_val_score(nb_model, train_ind, train_dep, cv=10, scoring='f1_macro')
print("Naive Bayes F1 Score (10-fold cross-validation): {:.2f}".format(nb_scores.mean()))

dt_scores = cross_val_score(dt_model, train_ind, train_dep, cv=10, scoring='f1_macro')
print("Decision Tree F1 Score (10-fold cross-validation): {:.2f}".format(dt_scores.mean()))


rf_scores = cross_val_score(rf_model, train_ind, train_dep, cv=10, scoring='f1_macro')
print("Random Forest F1 Score (10-fold cross-validation): {:.2f}".format(rf_scores.mean()))

'''










