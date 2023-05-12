# Import required libraries
import pandas as pd
import csv
import json
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

# Collecting all files and cleaning up None values
review_path = "C:/Users/rafae/Documents/Python/cs_349/pessoal/devided_dataset_v2/devided_dataset_v2/CDs_and_Vinyl/train/review_training.json"
product_path = "C:/Users/rafae/Documents/Python/cs_349/pessoal/devided_dataset_v2/devided_dataset_v2/CDs_and_Vinyl/train/product_training.json"
train_product_data = pd.read_json(product_path)
train_review_data = pd.read_json(review_path)

# Select 10 products just to test
train_product_data = train_product_data.iloc[:10, :]

# Example text corpus
id_vector_awesome = train_product_data[train_product_data["awesomeness"] == 1]['asin']
df_awesome_corpus = train_review_data.loc[train_review_data['asin'].isin(id_vector_awesome), ['asin', 'summary']].reset_index(drop=True)
# print(df_awesome.head(20))
awesome_corpus = list(df_awesome_corpus.loc[:, 'summary'])
# print(awesome_corpus)

id_vector_notawesome = train_product_data[train_product_data["awesomeness"] == 0]['asin']
df_notawesome_corpus = train_review_data.loc[train_review_data['asin'].isin(id_vector_notawesome), ['asin', 'summary']].reset_index(drop=True)
# print(df_notawesome.head(20))
notawesome_corpus = list(df_notawesome_corpus.loc[:, 'summary'])
# print(notawesome_corpus)

# Create a CountVectorizer object
vectorizer_awesome = TfidfVectorizer(stop_words='english')
vectorizer_notawesome = TfidfVectorizer(stop_words='english')

# Fit the vectorizer to the corpus and transform the corpus into a document-term matrix
train_tfidf_awesome = vectorizer_awesome.fit_transform(awesome_corpus)
train_dtfidf_notawesome = vectorizer_notawesome.fit_transform(notawesome_corpus)

# Get the vocabulary
vocabulary_awesome = vectorizer_awesome.get_feature_names()
vocabulary_notawesome = vectorizer_notawesome.get_feature_names()

# Transforming document-term matrix to a dataframe
awesome_df = pd.DataFrame(train_tfidf_awesome.toarray(), columns=vocabulary_awesome)
notawesome_df = pd.DataFrame(train_dtfidf_notawesome.toarray(), columns=vocabulary_notawesome)

test_review = [df_awesome_corpus.loc[0, 'summary']]
print(type(test_review))

# transform the testing reviews into TF-IDF scores using the same vectorizer
test_tfidf_awesome = vectorizer_awesome.transform(test_review)
test_tfidf_notawesome = vectorizer_notawesome.transform(test_review)

# Transforming document-term matrix to a dataframe
test_tfidf_awesome_df = pd.DataFrame(test_tfidf_awesome.toarray(), columns=vocabulary_awesome)
# print(test_tfidf_awesome_df)
test_tfidf_notawesome_df = pd.DataFrame(test_tfidf_notawesome.toarray(), columns=vocabulary_notawesome)
# print(test_tfidf_notawesome_df)

# Sum the columns together and create a new column with the results
test_tfidf_awesome_df['col_sum'] = test_tfidf_awesome_df.sum(axis=1)
# print(test_tfidf_awesome_df)
test_tfidf_notawesome_df['col_sum'] = test_tfidf_notawesome_df.sum(axis=1)
# print(test_tfidf_notawesome_df)

# Adding asin back to the dataframe
# awesome_df['asin'] = df_awesome_corpus['asin'].copy()
# notawesome_df['asin'] = df_notawesome_corpus['asin'].copy()

# group by the product_id column and compute the mean of the value column for each group
# result_awesome = awesome_df.groupby('asin')['col_sum'].mean()
# result_notawesome = notawesome_df.groupby('asin')['col_sum'].mean()

# print(len(result_awesome))
# print(len(result_notawesome))

# Print the document-term matrix and the vocabulary
# print(awesome_df)
# print(notawesome_df)
# print(result_awesome)
# print(result_notawesome)
# print(vocabulary_awesome)
# print(vocabulary_notawesome)