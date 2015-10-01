'''
MACHINE LEARNING WITH KNN
'''
import pandas as pd


# read in the iris data
from sklearn.datasets import load_iris
iris = load_iris()


# create X (features) and y (response)
data = pd.read_csv('https://raw.githubusercontent.com/sinanuozdemir/SF_DAT_17/master/data/iris.csv')

X, y = data.drop('species', axis = 1), data['species']
X.shape
y.shape



# predict y with KNN
from sklearn.neighbors import KNeighborsClassifier  # import class

knn = KNeighborsClassifier(n_neighbors=1)           # instantiate the estimator

knn.fit(X, y)                                       # fit with data

knn.predict([3, 5, 4, 2])                           # predict for a new observation


# predict for multiple observations at once
X_new = [[3, 5, 4, 2], [3, 5, 2, 2]]
knn.predict(X_new)

# try a different value of K
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X, y)
knn.predict(X_new)              # predictions
knn.predict_proba(X_new)        # predicted probabilities
knn.kneighbors([3, 5, 4, 2])    # distances to nearest neighbors (and identities)

# compute the accuracy for K=5 and K=1

# K = 5
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X, y)
knn.score(X, y)
# the score function will return the accuracy of your prediction
# the number of correct prepdictions / the number of rows


# K = 1
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X, y)
knn.score(X, y)

