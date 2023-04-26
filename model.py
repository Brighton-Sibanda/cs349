# Importing required libraries
import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# Reading training data from CSV file
feature_vector_path = "feature_vector.csv"
feature_vector = pd.read_csv(feature_vector_path)

# Separating features and target variables in training data
train_ind = feature_vector.iloc[:, 1:-1]
train_dep = feature_vector.iloc[:, -1]

# Creating a Multinomial Naive Bayes model
nb_model = MultinomialNB()

# Training the model
nb_model.fit(train_ind, train_dep)

# Predicting the classification of training data
y_pred = nb_model.predict(train_ind)

# Calculating the accuracy of the model on training data
accuracy_train = accuracy_score(train_dep, y_pred)
print("Training Accuracy:", accuracy_train)