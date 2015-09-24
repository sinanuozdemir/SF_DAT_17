import pandas as pd


'''
Part 1: UFO

'''

ufo = pd.read_csv('https://raw.githubusercontent.com/sinanuozdemir/SF_DAT_17/master/data/ufo.csv')   # can also read csvs directly from the web!



# 1. change the column names so that each name has no spaces
#           and all lower case

# 2. Show a bar chart of all shapes reported

# 3. Show a dataframe that only displays the reportings from Utah

# 4. Show a dataframe that only displays the reportings from Texas

# 5. Show a dataframe that only displays the reportings from Utah OR Texas

# 6. Which shape is reported most often?

# 7. Plot number of sightings per day in 2014 (days should be in order!)



'''
Part 2: IRIS

'''


iris = pd.read_csv('https://raw.githubusercontent.com/sinanuozdemir/SF_DAT_17/master/data/iris.csv')   # can also read csvs directly from the web!

# 1. Show the mean petal length by flower species

# 2. Show the mean sepal width by flower species

# 3. Use the groupby to show both #1 and #2 in one dataframe

# 4. Create a scatter plot plotting petal length against petal width
#    Use the color_flowers function to 

# 5. Show flowers with sepal length over 5 and petal length under 1.5

# 6. Show setosa flowers with petal width of exactly 0.2

# 7. Write a function to predict the species for each observation


def classify_iris(data):
    if False:
        return 'Iris-setosa'
    elif False:
        return 'Iris-versicolor'
    else:
        return 'Iris-virginica'

# example use: 
# classify_iris([0,3,2.1,3.2]) == 'Iris-virginica'
# assume the order is the same as the dataframe, so:
# [sepal_length', 'sepal_width', 'petal_length', 'petal_width']


# make predictions and store as preds
preds = iris.drop('species', axis=1).apply(classify_iris, axis = 1)


preds


# test your function: compute accuracy of your prediction
(preds == iris['species']).sum() / float(iris.shape[0])


'''
Part 3: FIFA GOALS

'''

goals = pd.read_csv('https://raw.githubusercontent.com/sinanuozdemir/SF_DAT_17/master/data/fifa_goals.csv')
# removing '+' from minute and turning them into ints
goals.minute = goals.minute.apply(lambda x: int(x.replace('+','')))


goals.head()


# 1. Show goals scored in the first 5 minutes of a game


# 2. Show goals scored after the regulation 90 minutes is over


# 3. Show the top scoring players


# 4. Show a histogram of minutes with 20 bins

# 5. Show a histogram of the number of goals scored by players

