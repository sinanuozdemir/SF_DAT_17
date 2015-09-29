'''
Our KNN algorithm

'''
# imports go here
import pandas as pd
import numpy as np

'''
Part 1: Setting it all up

'''

# we will need a euclidean_distance_algorithm that takes in
# two numpy arrays, and calculates the 
# euclidean distance between them

def euclidean_distance(np1, np2):
    return np.linalg.norm(np1-np2)


'''
Bring in the iris data from the web

iris_data ==
    2D numpy array of the four predictors of iris
        plus the species
'''

iris_data = pd.read_csv('https://raw.githubusercontent.com/sinanuozdemir/SF_DAT_15/master/data/iris.csv')
# iris_data is a dataframe, but let's turn it into
# a 2D numpy array
# Hint: use .values to turn a dataframe into a 2d array

iris_data = iris_data.values

# Question: in terms of machine learning:
#   a. the first four columns are called what?
#   b. the species column is called what?

iris_data
'''
array([[5.1, 3.5, 1.4, 0.2, 'Iris-setosa'],
       [4.9, 3.0, 1.4, 0.2, 'Iris-setosa'],
       [4.7, 3.2, 1.3, 0.2, 'Iris-setosa'],
       ... ...
       [6.2, 3.4, 5.4, 2.3, 'Iris-virginica'],
       [5.9, 3.0, 5.1, 1.8, 'Iris-virginica']], dtype=object)
'''


    
'''
Part 2: Predictions

Before we jump into making a general function,
let's try to predict 

unknown = [ 6.3,  3.1 ,  5.1,  2.4] with 3 neighbors
'''

# define our variables
unknown = [ 6.3,  3.1 ,  5.1,  2.4]
k = 3

# Make a a list "distances" consisting of tuples
# Each tuple should be
# (euc_distance(unknown, data_point), species)
# for each data_point in iris_data
distances = [(euclidean_distance(unknown, \
row[:-1]),row[-1]) for row in iris_data]

# OR

distances = []
for row in iris_data:
    flower_data = row[:-1]
    distance = euclidean_distance(unknown, flower_data)
    distances.append((distance, row[-1]))

distances
'''
== [(4.4866468548349108, 'setosa'),
 (4.5276925690687078, 'setosa'),
 (4.6743983570080969, 'setosa'),
 ...
 (0.44721359549995821, 'virginica'),
 (0.72801098892805138, 'virginica')]

'''

# Grab the nearest k neighbors

# Now we need to take the most frequently occuring flower
# in those k neighbors
# To do this, we will use the collections module

# given a list l, this code will spit back the mode
from collections import Counter
l = [1,2,3,4,3,2,2,5,8,2,2,2,5,9,2,2,5,5,3,2]
Counter(l).most_common(1)[0][0] # == 2

# use it to find the most frequent occuring flower in nearest
# note that the species is in the 1th index
prediction = Counter([n[1] for n in nearest]).\
most_common(1)[0][0]
    
'''
Time to put it in a function so we 
can apply the prediction
to each row in our data set!
'''
    
# will default to 3 neighbors
def predict(unknown, k = 3):
    '''
    Input:
        unknown  == four attributes of an unknown flower
        k        == the number of neighbors used
    Output:
        A prediction of the species of flower (str)
    '''
    distances = [(euclidean_distance(unknown, row[:-1]),row[-1]) for row in iris_data]
    nearest = sorted(distances)[:k]

    return Counter([n[1] for n in nearest]).most_common(1)[0][0]
    

predict([ 5.8,  2.7,  5.1,  1.9]) # == 'virginica'



'''
Putting it all together
'''

# predict each row
# Note I use row[-1] because the last element of each row 
# is the actual species
predictions = np.array([predict(row[:4]) for row in iris_data])

# this is the last column of the iris_data
actual = np.array([row[-1] for row in iris_data])

# accuracy of the model
np.mean(predictions == actual)



# now with k == 5
predictions = np.array([predict(row[:4], k = 5) for row in iris_data])


# this is the last column of the iris_data
actual = np.array([row[-1] for row in iris_data])

# accuracy of the model
np.mean(predictions == actual)




# now with k == 2
predictions = np.array([predict(row[:4], k = 2) for row in iris_data])


# this is the last column of the iris_data
actual = np.array([row[-1] for row in iris_data])

# accuracy of the model
np.mean(predictions == actual)



# now with k == 1
predictions = np.array([predict(row[:4], k = 1) for row in iris_data])


# this is the last column of the iris_data
actual = np.array([row[-1] for row in iris_data])

# accuracy of the model
np.mean(predictions == actual)

# only two neighbors is the best so far!

