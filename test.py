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

product_data = product_data.iloc[:10, :]

# Example text corpus
id_vector_awesome = product_data[product_data["awesomeness"] == 1]['asin']
df_awesome = review_data[review_data['asin'].isin(id_vector_awesome)]['summary']
#print(df_awesome.head())
awesome_corpus = list(df_awesome)
print(awesome_corpus)

id_vector_notawesome = product_data[product_data["awesomeness"] == 0]['asin']
df_notawesome = review_data[review_data['asin'].isin(id_vector_notawesome)]['summary']
#print(df_notawesome.head())
notawesome_corpus = list(df_notawesome)
print(notawesome_corpus)

# Create a CountVectorizer object
vectorizer_awesome = TfidfVectorizer(stop_words='english')
vectorizer_notawesome = TfidfVectorizer(stop_words='english')

# Fit the vectorizer to the corpus and transform the corpus into a document-term matrix
doc_term_matrix_awesome = vectorizer_awesome.fit_transform(awesome_corpus)
doc_term_matrix_notawesome = vectorizer_notawesome.fit_transform(notawesome_corpus)

# Get the vocabulary
vocabulary_awesome = vectorizer_awesome.get_feature_names()
vocabulary_notawesome = vectorizer_notawesome.get_feature_names()

awesome = pd.DataFrame(doc_term_matrix_awesome.toarray(), columns=vectorizer_awesome.get_feature_names())
notawesome = pd.DataFrame(doc_term_matrix_notawesome.toarray(), columns=vectorizer_notawesome.get_feature_names())

# Print the document-term matrix and the vocabulary
print(doc_term_matrix_awesome.toarray())
print(doc_term_matrix_notawesome.toarray())
print(vocabulary_awesome)
print(vocabulary_notawesome)