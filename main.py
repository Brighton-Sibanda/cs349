from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import warnings
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV



warnings.filterwarnings("ignore")


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
test_review_path = "devided_dataset_v2/CDs_and_Vinyl/test2/review_test.json"
test_product_path = "devided_dataset_v2/CDs_and_Vinyl/test2/product_test.json"
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
#product_data = product_data.iloc[:30, :]


def get_TFIDF(text):
    
    ''' This function calculates the TFIDF score of text, based on the two corpuses awesome and not awesome'''
    
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
    '''This function determines whether text is relatively likely more awesome than it is liekly not awesome; '''
    return df[text1].mean() > df[text2].mean()


def is_negative(df, text1, text2):
    '''This function determines whether text is relatively likely less awesome than it is liekly not awesome; '''
    return df[text1].mean() < df[text2].mean()


def get_vote_score(df):
    
    '''Calculates the effect of votes based on TFIDF scores of text '''
    
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
    on the relative time it was sent, and whether is is liekly awesome than not awesome
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
    """ function to count ratio of likely awesome reviews with images and 
    likely not awesome reviews with images"""
    
    df_new = df[df["image"]==True]
    pos1 = is_positive(df_new, 'aw_rt', 'naw_rt')
    neg1 = is_negative(df_new, 'aw_rt', 'naw_rt')
     
    
    pos2 = is_positive(df_new, 'aw_s', 'naw_s')
    neg2 = is_negative(df_new, 'aw_s', 'naw_s')
    
    
    return ((pos1 + pos2)+1) / ((neg1 + neg2) + 1)

def num_verified(df):

    """function to count ratio of likely awesome verified and likely not awesome verified reviews"""

    df_new = df[df["verified"]==True]
    pos1 = is_positive(df_new, 'aw_rt', 'naw_rt')
    neg1 = is_negative(df_new, 'aw_rt', 'naw_rt')
     
    
    pos2 = is_positive(df_new, 'aw_s', 'naw_s')
    neg2 = is_negative(df_new, 'aw_s', 'naw_s')
    
    
    return ((pos1 + pos2) + 1) / ( (neg1 + neg2) + 1)

    
feature_vector = pd.DataFrame({"aw_rt":[], "naw_rt":[], "aw_s":[], "naw_s":[],"vote_score":[], "image_score":[], "verified":[], "time_score":[]})
iDs = list(product_data['asin'])



def make_feature_vector(iDs, feature_vector, review_data, text):
    
    """loop through the products and construct the row with the feature vector 
    values"""
    
    k = 0
    for i in iDs:
        if (k % 1000 == 0):
            print("now finished with " + str(k) + " products, for " + text)
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

print ("model is now starting training")
train_feature_vector = make_feature_vector(iDs, feature_vector, review_data, "training")
train_feature_vector["awesomeness"] = list(product_data["awesomeness"])



#NOW TRAINING
X = train_feature_vector.iloc[:, :-1]
y = train_feature_vector.iloc[:, -1]
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the parameter grid
param_grid_lr = {
    'penalty': ['l1', 'l2', 'elasticnet', 'none'],
    'C': [0.1, 1, 10],
    'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
    'max_iter': [100, 200, 300],
    'multi_class': ['ovr', 'multinomial']
}


lr_model = LogisticRegression()
lr_grid_search = GridSearchCV(lr_model, param_grid_lr, cv=5)
lr_grid_search.fit(X, y)
lr_model = LogisticRegression(**lr_grid_search.best_params_)
lr_model.fit(X, y)


#train_feature_vector.to_csv("feature_vectorv2.csv")
print ("Done with training, now onto test data \n \n")


#Now for testing
iDs = list(test_product_data['asin'])
feature_vector_2 = pd.DataFrame({"num_pos":[], "num_neg":[], "vote_score":[], "pos_image_count":[], "neg_image_count":[], "pos_verified_count":[], "neg_verified_count":[], "pos_time_score":[], "neg_time_score":[]})
test_feature_vector = make_feature_vector(iDs, feature_vector_2, test_review_data, "testing")
predicted_class = lr_model.predict(test_feature_vector)

final_json = test_product_data
final_json["awesomeness"] = predicted_class
final_json.to_json("predictions.json")



#running metrics code; commented for time here
print("Now running metrics,, they've already been done. Comment out the code at the bottom if you wish to double check results. Thank you.")


"""

from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import cross_val_score



feature_vector = train_feature_vector

# Separating features and target variables in training data
train_ind = feature_vector.iloc[:, -1]
train_dep = feature_vector.iloc[:, -1]

# Define the parameter grid for NB
param_grid_nb = {
    'alpha': [0.01, 0.1, 1.0],
    'fit_prior': [True, False],
    'class_prior': [None, [0.4, 0.6], [0.3, 0.5, 0.2]]
}

# Define parameter grid for Knn
param_grid_knn = {
    'n_neighbors': [3, 5, 7],
    'weights': ['uniform', 'distance'],
    'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
}

# Define parameter grid for dt
param_grid_dt = {'criterion': ['gini', 'entropy'],
    'splitter': ['best', 'random'],
    'max_depth': [None, 5, 10],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['auto', 'sqrt', 'log2', None]
}

# Define parameter grid for svm
param_grid_svm = {'C': [0.1, 1, 10, 100],
    'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
    'gamma': [0.01, 0.1, 1, 'scale'],
    'degree': [2, 3, 4]
}

# Define parameter grid for lr
param_grid_lr = {
    'penalty': ['l1', 'l2', 'elasticnet', 'none'],
    'C': [0.1, 1, 10],
    'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
    'max_iter': [100, 200, 300],
    'multi_class': ['ovr', 'multinomial']
}

# Define parameter grid for rf
param_grid_rf = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2'],
    'bootstrap': [True, False]
}

# Define the parameter grid for Gradient Boosting
param_grid_gb = {
    'learning_rate': [0.1, 0.01],
    'n_estimators': [100, 200],
    'max_depth': [3, 5],
    'min_samples_split': [2, 4] 
}

# Creating models
nb_model = MultinomialNB()
knn_model = KNeighborsClassifier()
dt_model = DecisionTreeClassifier()
svm_model = SVC()
lr_model = LogisticRegression()
rf_model = RandomForestClassifier()
gb_model = GradientBoostingClassifier()



# Calculating precision, recall, and F1 score of each model on training data
# Performing Hyperparameter Optimization MultinomialNB - perform grid search cross-validation
nb_grid_search = GridSearchCV(nb_model, param_grid_nb, cv=5)
nb_grid_search.fit(train_ind, train_dep)
nb_model = MultinomialNB(**nb_grid_search.best_params_)
nb_model.fit(train_ind, train_dep)
nb_y_pred = nb_model.predict(train_ind)

nb_precision = precision_score(train_dep, nb_y_pred, average='macro')
nb_recall = recall_score(train_dep, nb_y_pred, average='macro')
nb_f1 = f1_score(train_dep, nb_y_pred, average='macro')
print("Multinomial Naive Bayes Precision: {:.2f}".format(nb_precision))
print("Multinomial Naive Bayes Recall: {:.2f}".format(nb_recall))
print("Multinomial Naive Bayes F1 Score: {:.2f}".format(nb_f1))


knn_grid_search = GridSearchCV(knn_model, param_grid_knn, cv=5)
knn_grid_search.fit(train_ind, train_dep)
knn_model = KNeighborsClassifier(**knn_grid_search.best_params_)
knn_model.fit(train_ind, train_dep)
knn_y_pred = knn_model.predict(train_ind)

knn_precision = precision_score(train_dep, knn_y_pred, average='macro')
knn_recall = recall_score(train_dep, knn_y_pred, average='macro')
knn_f1 = f1_score(train_dep, knn_y_pred, average='macro')
print("K-Nearest Neighbor Precision: {:.2f}".format(knn_precision))
print("K-Nearest Neighbor Recall: {:.2f}".format(knn_recall))
print("K-Nearest Neighbor F1 Score: {:.2f}".format(knn_f1))

dt_grid_search = GridSearchCV(dt_model, param_grid_dt, cv=5)
dt_grid_search.fit(train_ind, train_dep)
dt_model = DecisionTreeClassifier(**dt_grid_search.best_params_)
dt_model.fit(train_ind, train_dep)
dt_y_pred = dt_model.predict(train_ind)
dt_precision = precision_score(train_dep, dt_y_pred, average='macro')
dt_recall = recall_score(train_dep, dt_y_pred, average='macro')
dt_f1 = f1_score(train_dep, dt_y_pred, average='macro')
print("Decision Tree Precision: {:.2f}".format(dt_precision))
print("Decision Tree Recall: {:.2f}".format(dt_recall))
print("Decision Tree F1 Score: {:.2f}".format(dt_f1))



lr_grid_search = GridSearchCV(lr_model, param_grid_lr, cv=5)
lr_grid_search.fit(train_ind, train_dep)
lr_model = LogisticRegression(**lr_grid_search.best_params_)
lr_model.fit(train_ind, train_dep)
lr_y_pred = lr_model.predict(train_ind)
lr_precision = precision_score(train_dep, lr_y_pred, average='macro')
lr_recall = recall_score(train_dep, lr_y_pred, average='macro')
lr_f1 = f1_score(train_dep, lr_y_pred, average='macro')
print("Logistic Regression Precision: {:.2f}".format(lr_precision))
print("Logistic Regression Recall: {:.2f}".format(lr_recall))
print("Logistic Regression F1 Score: {:.2f}".format(lr_f1))


rf_grid_search = GridSearchCV(rf_model, param_grid_rf, cv=5)
rf_grid_search.fit(train_ind, train_dep)
rf_model = RandomForestClassifier(**rf_grid_search.best_params_)
rf_model.fit(train_ind, train_dep)
rf_y_pred = rf_model.predict(train_ind)

rf_precision = precision_score(train_dep, rf_y_pred, average='macro')
rf_recall = recall_score(train_dep, rf_y_pred, average='macro')
rf_f1 = f1_score(train_dep, rf_y_pred, average='macro')
print("Random Forest Precision: {:.2f}".format(rf_precision))
print("Random Forest Recall: {:.2f}".format(rf_recall))
print("Random Forest F1 Score: {:.2f}".format(rf_f1))


gb_grid_search = GridSearchCV(gb_model, param_grid_gb, cv=5)
gb_grid_search.fit(train_ind, train_dep)
gb_model = GradientBoostingClassifier(**gb_grid_search.best_params_)
gb_model.fit(train_ind, train_dep)
gb_y_pred = gb_model.predict(train_ind)
gb_precision = precision_score(train_dep, gb_y_pred, average='macro')
gb_recall = recall_score(train_dep, gb_y_pred, average='macro')
gb_f1 = f1_score(train_dep, gb_y_pred, average='macro')
print("Gradient Boosting Precision: {:.2f}".format(gb_precision))
print("Gradient Boosting Recall: {:.2f}".format(gb_recall))
print("Gradient Boosting F1 Score: {:.2f}".format(gb_f1))

gb_scores = cross_val_score(gb_model, train_ind, train_dep, cv=10, scoring='f1_macro')
gb_2scores = cross_val_score(gb_model, train_ind, train_dep, cv=10, scoring='precision')
gb_3scores = cross_val_score(gb_model, train_ind, train_dep, cv=10, scoring='accuracy')
gb_4scores = cross_val_score(gb_model, train_ind, train_dep, cv=10, scoring='recall')
print("Gradient Boosting F1 Score (10-fold cross-validation): {:.2f}".format(gb_scores.mean()))
print("Gradient Boosting precision Score (10-fold cross-validation): {:.2f}".format(gb_2scores.mean()))
print("Gradient Boosting accuracy Score (10-fold cross-validation): {:.2f}".format(gb_3scores.mean()))
print("Gradient Boosting recall Score (10-fold cross-validation): {:.2f}".format(gb_4scores.mean()))



knn_scores = cross_val_score(knn_model, train_ind, train_dep, cv=10, scoring='f1_macro')
print("k nearest neighbor f1 Score (10-fold cross-validation): {:.2f}".format(knn_scores.mean()))




lr_scores = cross_val_score(lr_model, train_ind, train_dep, cv=10, scoring='f1_macro')
lr_2scores = cross_val_score(lr_model, train_ind, train_dep, cv=10, scoring='precision')
lr_3scores = cross_val_score(lr_model, train_ind, train_dep, cv=10, scoring='accuracy')
lr_4scores = cross_val_score(lr_model, train_ind, train_dep, cv=10, scoring='recall')
print("Logistic Regression F1 Score (10-fold cross-validation): {:.2f}".format(lr_scores.mean()))
print("Logistic Regression precision  Score (10-fold cross-validation): {:.2f}".format(lr_2scores.mean()))
print("Logistic Regression accuracy Score (10-fold cross-validation): {:.2f}".format(lr_3scores.mean()))
print("Logistic Regression recall Score (10-fold cross-validation): {:.2f}".format(lr_4scores.mean()))



nb_scores = cross_val_score(nb_model, train_ind, train_dep, cv=10, scoring='f1_macro')
print("Naive Bayes F1 Score (10-fold cross-validation): {:.2f}".format(nb_scores.mean()))

dt_scores = cross_val_score(dt_model, train_ind, train_dep, cv=10, scoring='f1_macro')
print("Decision Tree F1 Score (10-fold cross-validation): {:.2f}".format(dt_scores.mean()))


rf_scores = cross_val_score(rf_model, train_ind, train_dep, cv=10, scoring='f1_macro')
print("Random Forest F1 Score (10-fold cross-validation): {:.2f}".format(rf_scores.mean()))


print("\n \n")
print ("Support vector machine takes a very long time to run; disregarding it as an option")
''' 
svm_grid_search = GridSearchCV(svm_model, param_grid_svm, cv=5)
svm_grid_search.fit(train_ind, train_dep)
svm_model = SVC(**svm_grid_search.best_params_)
svm_model.fit(train_ind, train_dep)
svm_y_pred = svm_model.predict(train_ind)
svm_precision = precision_score(train_dep, svm_y_pred, average='macro')
svm_recall = recall_score(train_dep, svm_y_pred, average='macro')
svm_f1 = f1_score(train_dep, svm_y_pred, average='macro')
print("Support Vector Machine Precision: {:.2f}".format(svm_precision))
print("Support Vector Machine Recall: {:.2f}".format(svm_recall))
print("Support Vector Machine F1 Score: {:.2f}".format(svm_f1))

svm_scores = cross_val_score(svm_model, train_ind, train_dep, cv=10, scoring='f1_macro')
print("Support Vector Machine F1 Score (10-fold cross-validation): {:.2f}".format(svm_scores.mean()))'''



"""





