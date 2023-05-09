# Import required libraries
import pandas as pd
import csv
import json
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

# Collecting all files and cleaning up None values
review_path = "devided_dataset_v2/CDs_and_Vinyl/train/review_training.json"
product_path = "devided_dataset_v2/CDs_and_Vinyl/train/product_training.json"
product_data = pd.read_json(product_path)
review_data = pd.read_json(review_path)

# Select 10 products just to test
product_data = product_data.iloc[:10, :]

# Example text corpus
id_vector_awesome = product_data[product_data["awesomeness"] == 1]['asin']
df_awesome = review_data.loc[review_data['asin'].isin(id_vector_awesome), ['asin', 'summary']].reset_index(drop=True)
# print(df_awesome.head(20))
awesome_corpus = list(df_awesome.loc[:, 'summary'])
# print(awesome_corpus)

id_vector_notawesome = product_data[product_data["awesomeness"] == 0]['asin']
df_notawesome = review_data.loc[review_data['asin'].isin(id_vector_notawesome), ['asin', 'summary']].reset_index(drop=True)
# print(df_notawesome.head(20))
notawesome_corpus = list(df_notawesome.loc[:, 'summary'])
# print(notawesome_corpus)

# Create a CountVectorizer object
vectorizer_awesome = TfidfVectorizer(stop_words='english')
vectorizer_notawesome = TfidfVectorizer(stop_words='english')

# Fit the vectorizer to the corpus and transform the corpus into a document-term matrix
doc_term_matrix_awesome = vectorizer_awesome.fit_transform(awesome_corpus)
doc_term_matrix_notawesome = vectorizer_notawesome.fit_transform(notawesome_corpus)

# Get the vocabulary
vocabulary_awesome = vectorizer_awesome.get_feature_names()
vocabulary_notawesome = vectorizer_notawesome.get_feature_names()

# Transforming document-term matrix to a dataframe
awesome_df = pd.DataFrame(doc_term_matrix_awesome.toarray(), columns=vectorizer_awesome.get_feature_names())
notawesome_df = pd.DataFrame(doc_term_matrix_notawesome.toarray(), columns=vectorizer_notawesome.get_feature_names())
print(len(awesome_df))
print(len(notawesome_df))

# sum the columns together and create a new column with the results
awesome_df['col_sum'] = awesome_df.sum(axis=1)
notawesome_df['col_sum'] = notawesome_df.sum(axis=1)

# Adding asin back to the dataframe
awesome_df['asin'] = df_awesome['asin'].copy()
notawesome_df['asin'] = df_notawesome['asin'].copy()

# group by the product_id column and compute the mean of the value column for each group
result_awesome = awesome_df.groupby('asin')['col_sum'].mean()
result_notawesome = notawesome_df.groupby('asin')['col_sum'].mean()

# Print the document-term matrix and the vocabulary
# print(awesome_df)
# print(notawesome_df)
# print(result_awesome)
# print(result_notawesome)
# print(vocabulary_awesome)
# print(vocabulary_notawesome)