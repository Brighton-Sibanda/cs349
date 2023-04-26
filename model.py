# Importing required libraries
import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score

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

# Calculating precision, recall, and F1 score of the model on test data
precision = precision_score(train_dep, y_pred, average='macro')
recall = recall_score(train_dep, y_pred, average='macro')
f1 = f1_score(train_dep, y_pred, average='macro')
print("Precision: {:.2f}".format(precision))
print("Recall: {:.2f}".format(recall))
print("F1 Score: {:.2f}".format(f1))

# Calculating the accuracy of the model on training data
accuracy_train = accuracy_score(train_dep, y_pred)
print("Training Accuracy:", accuracy_train)