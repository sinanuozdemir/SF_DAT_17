"""

Principal Component Analysis applied to the Iris dataset.

"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from sklearn.neighbors import KNeighborsClassifier  # import class
from sklearn.cross_validation import cross_val_score

from sklearn import decomposition
from sklearn import datasets

# Load in the data
iris = datasets.load_iris()
X = iris.data
y = iris.target
target_names = iris.target_names


# KNN with the original iris
knn = KNeighborsClassifier(n_neighbors=5)
cross_val_score(knn, X, y, cv=10, scoring='accuracy').mean()



#############################
### PCA with 2 components  ##
#############################


pca = decomposition.PCA(n_components=2)
X_r = pca.fit_transform(X)

plt.figure()
for c, i, target_name in zip("rgb", [0, 1, 2], target_names):
    plt.scatter(X_r[y == i, 0], X_r[y == i, 1], c=c, label=target_name)
plt.legend()
plt.title('PCA(2 components) of IRIS dataset')

X_transformedSK = pca.transform(X)
# only 2 columns!!

# KNN with PCAed data
knn = KNeighborsClassifier(n_neighbors=5)
cross_val_score(knn, X_transformedSK, y, cv=10, scoring='accuracy').mean()


X_reconstituted = pca.inverse_transform(X_transformedSK)
# Turn it back into its 4 column using only 2 principal components

plt.scatter(X[:,2], X[:,3])
plt.scatter(X_reconstituted[:,2], X_reconstituted[:,3])
# it is only looking at 2 dimensions of data!


#############################
### PCA with 3 components  ##
#############################





plt.cla()
pca = decomposition.PCA(n_components=3)
pca.fit(X)
X_3 = pca.transform(X)

X_3

# KNN with 3 components
knn = KNeighborsClassifier(n_neighbors=5)
cross_val_score(knn, X_3, y, cv=10, scoring='accuracy').mean()



X_reconstituted = pca.inverse_transform(X_3)

plt.scatter(X[:,2], X[:,3])
plt.scatter(X_reconstituted[:,2], X_reconstituted[:,3])



#############################
### choosing components  ####
#############################



pca = decomposition.PCA(n_components=4)
X_r = pca.fit_transform(X)

# Percentage of variance explained for each components
print('explained variance ratio (first two components): %s'
      % str(pca.explained_variance_ratio_))

plt.cla()
plt.plot(pca.explained_variance_ratio_)
plt.title('Variance explained by each principal component')
plt.ylabel(' % Variance Explained')
plt.xlabel('Principal component')

# 2 components is enough!!

