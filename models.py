# Importing required libraries
import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import cross_val_score

# Reading training data from CSV file
feature_vector_path = "feature_vector.csv"
feature_vector = pd.read_csv(feature_vector_path)

# Separating features and target variables in training data
train_ind = feature_vector.iloc[:, 2:-1]
train_dep = feature_vector.iloc[:, -1]

# Creating models
# nb_model = MultinomialNB()
# knn_model = KNeighborsClassifier()
dt_model = DecisionTreeClassifier()
# svm_model = SVC()
# lr_model = LogisticRegression()
rf_model = RandomForestClassifier()
# gb_model = GradientBoostingClassifier()

# Training the models
# nb_model.fit(train_ind, train_dep)
# knn_model.fit(train_ind, train_dep)
dt_model.fit(train_ind, train_dep)
# svm_model.fit(train_ind, train_dep)
# lr_model.fit(train_ind, train_dep)
rf_model.fit(train_ind, train_dep)
# gb_model.fit(train_ind, train_dep)

# Predicting the classification of training data for each model
# nb_y_pred = nb_model.predict(train_ind)
# knn_y_pred = knn_model.predict(train_ind)
dt_y_pred = dt_model.predict(train_ind)
# svm_y_pred = svm_model.predict(train_ind)
# lr_y_pred = lr_model.predict(train_ind)
rf_y_pred = rf_model.predict(train_ind)
# gb_y_pred = gb_model.predict(train_ind)

# Calculating precision, recall, and F1 score of each model on training data
# nb_precision = precision_score(train_dep, nb_y_pred, average='macro')
# nb_recall = recall_score(train_dep, nb_y_pred, average='macro')
# nb_f1 = f1_score(train_dep, nb_y_pred, average='macro')
# print("Multinomial Naive Bayes Precision: {:.2f}".format(nb_precision))
# print("Multinomial Naive Bayes Recall: {:.2f}".format(nb_recall))
# print("Multinomial Naive Bayes F1 Score: {:.2f}".format(nb_f1))

# knn_precision = precision_score(train_dep, knn_y_pred, average='macro')
# knn_recall = recall_score(train_dep, knn_y_pred, average='macro')
# knn_f1 = f1_score(train_dep, knn_y_pred, average='macro')
# print("K-Nearest Neighbor Precision: {:.2f}".format(knn_precision))
# print("K-Nearest Neighbor Recall: {:.2f}".format(knn_recall))
# print("K-Nearest Neighbor F1 Score: {:.2f}".format(knn_f1))

dt_precision = precision_score(train_dep, dt_y_pred, average='macro')
dt_recall = recall_score(train_dep, dt_y_pred, average='macro')
dt_f1 = f1_score(train_dep, dt_y_pred, average='macro')
print("Decision Tree Precision: {:.2f}".format(dt_precision))
print("Decision Tree Recall: {:.2f}".format(dt_recall))
print("Decision Tree F1 Score: {:.2f}".format(dt_f1))

# svm_precision = precision_score(train_dep, svm_y_pred, average='macro')
# svm_recall = recall_score(train_dep, svm_y_pred, average='macro')
# svm_f1 = f1_score(train_dep, svm_y_pred, average='macro')
# print("Support Vector Machine Precision: {:.2f}".format(svm_precision))
# print("Support Vector Machine Recall: {:.2f}".format(svm_recall))
# print("Support Vector Machine F1 Score: {:.2f}".format(svm_f1))

# lr_precision = precision_score(train_dep, lr_y_pred, average='macro')
# lr_recall = recall_score(train_dep, lr_y_pred, average='macro')
# lr_f1 = f1_score(train_dep, lr_y_pred, average='macro')
# print("Logistic Regression Precision: {:.2f}".format(lr_precision))
# print("Logistic Regression Recall: {:.2f}".format(lr_recall))
# print("Logistic Regression F1 Score: {:.2f}".format(lr_f1))

rf_precision = precision_score(train_dep, rf_y_pred, average='macro')
rf_recall = recall_score(train_dep, rf_y_pred, average='macro')
rf_f1 = f1_score(train_dep, rf_y_pred, average='macro')
print("Random Forest Precision: {:.2f}".format(rf_precision))
print("Random Forest Recall: {:.2f}".format(rf_recall))
print("Random Forest F1 Score: {:.2f}".format(rf_f1))

# gb_precision = precision_score(train_dep, gb_y_pred, average='macro')
# gb_recall = recall_score(train_dep, gb_y_pred, average='macro')
# gb_f1 = f1_score(train_dep, gb_y_pred, average='macro')
# print("Gradient Boosting Precision: {:.2f}".format(gb_precision))
# print("Gradient Boosting Recall: {:.2f}".format(gb_recall))
# print("Gradient Boosting F1 Score: {:.2f}".format(gb_f1))

dt_scores = cross_val_score(dt_model, train_ind, train_dep, cv=10, scoring='f1_macro')
print("Decision Tree F1 Score (10-fold cross-validation): {:.2f}".format(dt_scores.mean()))

rf_scores = cross_val_score(rf_model, train_ind, train_dep, cv=10, scoring='f1_macro')
print("Random Forest F1 Score (10-fold cross-validation): {:.2f}".format(rf_scores.mean()))