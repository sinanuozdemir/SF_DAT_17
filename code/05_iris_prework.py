'''
EXERCISE: "Human Learning" with iris data
'''

from sklearn.datasets import load_iris
import pandas as pd
import numpy as np

# load the famous iris data
iris = load_iris()

# what do you think these attributes represent?
iris.data
iris.data.shape
iris.feature_names
iris.target
iris.target_names

# intro to numpy
type(iris.data)


## PART 1: Read data into pandas and explore

iris.feature_names
# the feature_names are a bit messy, let's 
# clean them up. remove the (cm)
# at the end and replace any spaces with an underscore
# create a list called "features" that 
# holds the cleaned column names
features = [] # <FILL IN>


# read the iris data into pandas, with our refined column names
df = pd.DataFrame(iris.data, columns=features)


# create a list of species (should be 150 elements) 
# using iris.target and iris.target_names
# resulting list should only have the words "setosa", "versicolor", and "virginica"
'''
species ==  
['setosa',
 'setosa',
 'setosa',
 'setosa',
...
...
 'virginica',
 'virginica']

Hint: use the iris.target_names and iris.target arrays
'''
species = [] # <FILL IN>



# add the species list as a new DataFrame column
df['species'] = species


# explore data numerically, looking for differences between species
# try grouping by species and check out the different predictors
# explore data numerically, looking for differences between species
df.describe()
df.groupby('species').sepal_length.mean()
df.groupby('species')['sepal_length', 'sepal_width', 'petal_length', 'petal_width'].mean()
df.groupby('species').agg(np.mean)
df.groupby('species').agg([np.min, np.max])
df.groupby('species').describe()

'''
agg is a new function we haven't seen yet. It will
aggregate each column using specified lists of functions.
We have been using some of its shortcuts but using
agg allows us to put in many functions at a time

df.groupby('species').agg(np.mean)
==
df.groupby('species').mean()

BUT 
df.groupby('species').agg([np.min, np.max])

doesn't have a short form
'''

# explore data by sorting, looking for differences between species
df.sort_index(by='sepal_length').values
df.sort_index(by='sepal_width').values
df.sort_index(by='petal_length').values
df.sort_index(by='petal_width').values

# I used values in order to see all of the data at once
# without .values, a dataframe is returned


## PART 2: Write a function to predict the species for each observation

# create a dictionary so we can reference columns by name
# the key of the dictionary should be the species name
# the values should be the the strings index in the columns
# col_ix['sepal_length'] == 0
# col_ix['species'] == 4

col_ix = {} # <FILL IN>



# define function that takes in a row of data and returns a predicted species
def classify_iris(data):
    if False and False or False and False:
        return 'rose'
    elif False and (True or False):
        return 'lily'
    else:
        return 'I have no clue'

# make predictions and store as numpy array
preds = np.array([classify_iris(row) for row in df.values])


# calculate the accuracy of the predictions
np.mean(preds == df.species.values)