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
from sklearn.model_selection import GridSearchCV
import warnings
warnings.filterwarnings("ignore")


# Reading training data from CSV file
feature_vector_path = "feature_vectorv2.csv"
feature_vector = pd.read_csv(feature_vector_path)

# Separating features and target variables in training data
train_ind = feature_vector.iloc[:, 2:-1]
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
print("Logistic Regression F1 Score (10-fold cross-validation): {:.2f}".format(lr_scores.mean()))




nb_scores = cross_val_score(nb_model, train_ind, train_dep, cv=10, scoring='f1_macro')
print("Naive Bayes F1 Score (10-fold cross-validation): {:.2f}".format(nb_scores.mean()))

dt_scores = cross_val_score(dt_model, train_ind, train_dep, cv=10, scoring='f1_macro')
print("Decision Tree F1 Score (10-fold cross-validation): {:.2f}".format(dt_scores.mean()))


rf_scores = cross_val_score(rf_model, train_ind, train_dep, cv=10, scoring='f1_macro')
print("Random Forest F1 Score (10-fold cross-validation): {:.2f}".format(rf_scores.mean()))

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
print("Support Vector Machine F1 Score (10-fold cross-validation): {:.2f}".format(svm_scores.mean()))
