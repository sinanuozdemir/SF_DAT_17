from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import cross_val_score
from sklearn import svm
from sklearn import datasets
import matplotlib.pyplot as plt
olivetti = datasets.fetch_olivetti_faces()
X, y = olivetti.data, olivetti.target

X.shape
y.shape

# Try SVM
clf = svm.SVC()
clf.fit(X,y)
cross_val_score(clf, X, y, cv=5, scoring='accuracy').mean()

# Try Logistic Regression
logistic = LogisticRegression()
cross_val_score(logistic, X, y, cv=5, scoring='accuracy').mean()


plt.imshow(X[0].reshape(64, 64), cmap=plt.cm.gray_r)
plt.imshow(X[1].reshape(64, 64), cmap=plt.cm.gray_r)
y[0:2]

plt.imshow(X[200].reshape(64, 64), cmap=plt.cm.gray_r)
plt.imshow(X[201].reshape(64, 64), cmap=plt.cm.gray_r)
y[200:202]

from pybrain.datasets            import ClassificationDataSet
from pybrain.utilities           import percentError
from pybrain.tools.shortcuts     import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.structure.modules   import SoftmaxLayer
from numpy import ravel


# pybrain has its own data sample class that we must add
# our training and test set to
ds = ClassificationDataSet(4096, 1 , nb_classes=40)
for k in xrange(len(X)): 
    ds.addSample(ravel(X[k]),y[k])
    
# their equivalent of train test split
test_data, training_data = ds.splitWithProportion( 0.25 )


len(training_data.data['input'][0])

test_data

# pybrain's version of dummy variables
test_data._convertToOneOfMany( )
training_data._convertToOneOfMany( )


training_data['input']
training_data['target']

test_data.indim
test_data.outdim

# instantiate the model
fnn = buildNetwork( training_data.indim, 64, training_data.outdim, outclass=SoftmaxLayer )
trainer = BackpropTrainer( fnn, dataset=training_data, momentum=0.1, learningrate=0.01 , verbose=True, weightdecay=0.01) 

# change the number of eopchs to try to get better results!
trainer.trainEpochs (10)
print 'Percent Error on Test dataset: ' , \
        percentError( trainer.testOnClassData (
           dataset=test_data )
           , test_data['class'] )


