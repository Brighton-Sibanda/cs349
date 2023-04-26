# Importing required libraries
import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.model_selection import KFold
import numpy as np

# Reading training data from CSV file
feature_vector_path = "feature_vector.csv"
feature_vector = pd.read_csv(feature_vector_path)

# Separating features and target variables in training data
train_ind = feature_vector.iloc[:, 1:-1]
train_dep = feature_vector.iloc[:, -1]

# Creating a k-fold cross validation object with k=5
kf = KFold(n_splits=10, shuffle=True, random_state=42)

# Initializing an empty list to store the accuracy scores of each fold
accuracy_scores = []

# Looping over each fold of the cross validation object
for train_idx, val_idx in kf.split(train_ind):
    # Splitting the data into training and validation sets for this fold
    X_train, X_val = train_ind.iloc[train_idx], train_ind.iloc[val_idx]
    y_train, y_val = train_dep.iloc[train_idx], train_dep.iloc[val_idx]
    
    # Creating a new Naive Bayes model for this fold
    nb_model = MultinomialNB()

    # Training the model on the training data for this fold
    nb_model.fit(X_train, y_train)

    # Predicting the classification of validation data for this fold
    y_pred_val = nb_model.predict(X_val)

    # Calculating the accuracy of the model on the validation data for this fold
    accuracy_val = accuracy_score(y_val, y_pred_val)
    
    # Adding the accuracy score of this fold to the list
    accuracy_scores.append(accuracy_val)

# Calculating the mean and standard deviation of the accuracy scores across all folds
mean_accuracy = np.mean(accuracy_scores)
std_accuracy = np.std(accuracy_scores)

print("Mean Accuracy:", mean_accuracy)
print("Standard Deviation of Accuracy:", std_accuracy)

'''
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
'''