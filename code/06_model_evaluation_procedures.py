'''
CLASS: Model evaluation procedures
'''

import numpy as np
import matplotlib.pyplot as plt

# read in the iris data
from sklearn.datasets import load_iris
iris = load_iris()
X, y = iris.data, iris.target


## TRAIN AND TEST ON THE SAME DATA (OVERFITTING)

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X, y)
knn.score(X, y)


## TEST SET APPROACH

# understanding train_test_split
from sklearn.cross_validation import train_test_split
features = np.array([range(10), range(10, 20)]).T

features  # 2D array

response = ['even', 'odd'] * 5

response  # 1D array


features_train, features_test = train_test_split(features)

features_train  # 70% of the training set
features_test   # remaining 30% of the training set

features_train, features_test, response_train, response_test \
= train_test_split(features, response, random_state=1)
# the random_state allows us all to get the same random numbers

features_train  # 70% of the training set
features_test   # remaining 30% of the training set

response_train  #  70% of the responses, the SAME ones as features_train
response_test   # reamining 30%, SAME as features_test

# step 1: split data into training set and test set
X_train, X_test, y_train, y_test \
= train_test_split(X, y, random_state=4)

X.shape         # 150 rows

X_train.shape
X_test.shape
y_train.shape
y_test.shape

# steps 2 and 3: calculate test set error for K=1
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train, y_train)      # Note that I fit to the training
knn.score(X_test, y_test)      # and scored on the test set

# Suppose you watch a soccer game and memorize their patterns
# If you rewind the game and then were asked to predict
# the outcome, you'd know right?! That's why we train on one game
# and then get tested on our predictive ability on another game

# Overfitting

# step 4 (parameter tuning): calculate test set error for K=5
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
knn.score(X_test, y_test)

# steps 5 and 6: choose best model (K=5) and train on all data
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X, y)

# There are two types of data we will deal with in ML
# In sample and Out of sample data
# the in-sample data consists of our training and test sets
#   Note we know the answer to all of them!
# the out-of-sample data are data that we really haven't seen before
#   They're the reason we built it in the first time!

# step 7: make predictions on new ("out of sample") data
out_of_sample = [[5, 4, 3, 2], [4, 3, 2, 1]]
knn.predict(out_of_sample)

# verify that a different train/test split can result in a different test set error
X_train, X_test, y_train, y_test \
= train_test_split(X, y, random_state=1)
# I used a different random state so we all get the same results
# but different from the random_state = 4 from before!

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
knn.score(X_test, y_test)

# Using a training set and test set is so important
# Just as important is cross validation. Cross validation is
# Just using several different train test splits and 
#   averaging your results!

## CROSS-VALIDATION

# check CV score for K=1
from sklearn.cross_validation import cross_val_score
knn = KNeighborsClassifier(n_neighbors=1)
scores = cross_val_score(knn, X, y, cv=5, scoring='accuracy')

scores              # It ran a KNN 5 times!
# We are looking at the accuracy for each of the 5 splits

np.mean(scores)     # Average them together

# check CV score for K=5
knn = KNeighborsClassifier(n_neighbors=5)
scores = cross_val_score(knn, X, y, cv=5, scoring='accuracy')
scores
np.mean(scores)

# search for an optimal value of K
k_range = range(1, 30, 2)
scores = []
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores.append(np.mean(cross_val_score(knn, X, y, cv=5, scoring='accuracy')))
scores

# plot the K values (x-axis) versus the 5-fold CV score (y-axis)
plt.figure()
plt.plot(k_range, scores)

# automatic grid search for an optimal value of K
from sklearn.grid_search import GridSearchCV
knn = KNeighborsClassifier()
k_range = range(1, 30, 2)
param_grid = dict(n_neighbors=k_range)
grid = GridSearchCV(knn, param_grid, cv=5, scoring='accuracy')
grid.fit(X, y)

# this will check K=1, K=2, all the way up to 30,
# and then do cross validation on each one!
# thats 30 * 5 = 150 fits and scoring!

# check the results of the grid search
grid.grid_scores_
grid_mean_scores = [result[1] for result in grid.grid_scores_]

# plot the results
plt.figure()
plt.plot(k_range, grid_mean_scores)

grid.best_score_     # shows us the best score
grid.best_params_    # shows us the optimal parameters
grid.best_estimator_ # this is the actual model

