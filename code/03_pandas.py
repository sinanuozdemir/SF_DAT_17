'''
CLASS: Pandas for Data Exploration, Analysis, and Visualization

WHO alcohol consumption data:
    article: http://fivethirtyeight.com/datalab/dear-mona-followup-where-do-people-drink-the-most-beer-wine-and-spirits/    
    original data: https://github.com/fivethirtyeight/data/tree/master/alcohol-consumption
    files: drinks.csv (with additional 'continent' column)
'''

# imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


'''
Pandas Basics: Reading Files, Summarizing, Handling Missing Values, Filtering, Sorting
'''

# read in the CSV file from a URL
drinks = pd.read_csv('https://raw.githubusercontent.com/sinanuozdemir/SF_DAT_15/master/data/drinks.csv')

type(drinks)            # Use the type method to check python type

# examine the data
drinks                  # print the first 30 and last 30 rows
drinks.head()           # print the first 5 rows
drinks.tail()           # print the last 5 rows
drinks.describe()       # describe any numeric columns
drinks.info()           # concise summary
drinks.columns          # get series of column names
drinks.shape            # tuple of (#rows, #cols)

# find missing values in a DataFrame
drinks.isnull()         # DataFrame of booleans
drinks.isnull().sum()   # convert booleans to integers and add

# handling missing values
drinks.dropna()             # drop a row if ANY values are missing
drinks.fillna(value='NA')   # fill in missing values

# fix the original import
drinks = pd.read_csv('https://raw.githubusercontent.com/sinanuozdemir/SF_DAT_15/master/data/drinks.csv', na_filter=False)
drinks.isnull().sum()

# selecting a column ('Series')
drinks['continent']
drinks.continent            # equivalent
type(drinks.continent)      # Series if pandas equivalent to list

# summarizing a non-numeric column
drinks.continent.describe()
drinks.continent.value_counts()

# selecting multiple columns
drinks[['country', 'beer_servings']]
'''
note the double square bracket
the outer pair is used like in a python dictionary
    to select
the inner pair is a list!

so in all, the double use of square brackets is telling
the dataframe to select a list!
'''

my_cols = ['country', 'beer_servings']
drinks[my_cols]

# add a new column as a function of existing columns
drinks['total_servings'] = drinks.beer_servings +   drinks.spirit_servings + drinks.wine_servings
drinks.head()

# logical filtering and sorting
drinks[drinks.continent=='EU'] 

'''
How it works:
    drinks.continent=='EU' by itself returns a bunch
        of Trues and Falses
        
drinks.continent=='EU'

See?


when you wrap drinks around it with square brackets
you're telling the drinks dataframe to select
only those that are True, and not the False ones

drinks[drinks.continent=='EU']
'''

# North American countries with total servings
drinks[['country', 'total_servings']][drinks.continent=='NA']

# same thing, sorted by total_servings
drinks[['country', 'total_servings']][drinks.continent=='NA'].sort_index(by='total_servings')

# contries with wine servings over 300 and total liters over 12
drinks[drinks.wine_servings > 300][drinks.total_litres_of_pure_alcohol > 12]

# contries with more wine servings than beer servings
drinks[drinks.wine_servings > drinks.beer_servings]

# last 5 elements of the dataframe sorted by beer servings
drinks.sort_index(by='beer_servings').tail()

# average North American beer consumption
drinks.beer_servings[drinks.continent=='NA'].mean()
'''
Note the procedure:
drinks                                          Dataframe
drinks.beer_servings                            one column (Series)
drinks.beer_servings[drinks.continent=='NA']    logical filtering
drinks.beer_servings[drinks.continent=='NA'].mean() mean of that filtered column

'''

# average European beer consumption
drinks.beer_servings[drinks.continent=='EU'].mean()


'''
Split-Apply-Combine
'''

# for each continent, calculate mean beer servings
drinks.groupby('continent').beer_servings.mean()

# for each continent, count number of occurrences
drinks.groupby('continent').continent.count()
drinks.continent.value_counts()

# for each continent, calculate the min, max, and range for total servings
drinks.groupby('continent').total_servings.min()
drinks.groupby('continent').total_servings.max()


# We can apply any function using .apply
drinks.groupby('continent').total_servings.apply(lambda x: x.mean())    # mean

# note x here is an entire series
drinks.groupby('continent').total_servings.apply(lambda x: x.std())     # standard deviation

# What does this do?
drinks.groupby('continent').total_servings.apply(lambda x: x.max() - x.min())


'''
Plotting
'''

# bar plot of number of countries in each continent
drinks.continent.value_counts().plot(kind='bar', title='Countries per Continent')
plt.xlabel('Continent')
plt.ylabel('Count')
plt.show()

# bar plot of average number of beer servings by continent
drinks.groupby('continent').beer_servings.mean().plot(kind='bar')

# histogram of beer servings
drinks.beer_servings.hist(bins=20)

# grouped histogram of beer servings
drinks.beer_servings.hist(by=drinks.continent)

# stop and think, does this make sense

# same charts with the same scale for x and y axis
drinks.beer_servings.hist(by=drinks.continent, sharex=True, sharey=True)



# density plot of beer servings
drinks.beer_servings.plot(kind='density')

# same chart, with new x limit
drinks.beer_servings.plot(kind='density', xlim=(0,500))

# boxplot of beer servings by continent
drinks.boxplot(column='beer_servings', by='continent')

# scatterplot of beer servings versus wine servings
drinks.plot(x='beer_servings', y='wine_servings', kind='scatter', alpha=0.3)

# same scatterplot, except all European countries are colored red
colors = np.where(drinks.continent=='EU', 'r', 'b')

colors      # is a series of 'r' and 'b' that 
            # correspond to countries

'''
np.where is like a condensed if statement
it's like a list comprehension for pandas!

it will loop through drinks.continent which is a series
for each element:
    if it is "EU":
        make it 'r'
    else:
        make it 'b'

More in depth:
    drinks.continent=='EU' is a logical statement
        It will return a bunch of Trues and Falses
        and np.where makes the True ones 'r' and
        the False ones 'b'
        
        Recall logical filtering!

'''

# Side quest
np.where([True, False, False], 'a', 'b')

# 10 gold coins earned


drinks.plot(x='beer_servings', y='wine_servings', kind='scatter', c=colors)
# passing colors into the chart makes the european dots, red!


'''
Joining Data

MovieLens 100k data:
    main page: http://grouplens.org/datasets/movielens/
    data dictionary: http://files.grouplens.org/datasets/movielens/ml-100k-README.txt
    files: u.user, u.data, u.item



'''

# read 'u.data' into 'ratings'
r_cols = ['user_id', 'movie_id', 'rating', 'unix_timestamp']
ratings = pd.read_table('https://raw.githubusercontent.com/sinanuozdemir/SF_DAT_15/master/data/u.data', header=None, names=r_cols, sep='\t')

# read 'u.item' into 'movies'
m_cols = ['movie_id', 'title']
movies = pd.read_table('https://raw.githubusercontent.com/sinanuozdemir/SF_DAT_15/master/data/u.item', header=None, names=m_cols, sep='|', usecols=[0,1])

# merge 'movies' and 'ratings' (inner join on 'movie_id')
movies.head()
ratings.head()
movie_ratings = pd.merge(movies, ratings)
movie_ratings.head()


'''
Further Exploration
'''

# for each movie, count number of ratings
movie_ratings.title.value_counts()

# for each movie, calculate mean rating
movie_ratings.groupby('title').rating.mean().order(ascending=False)


'''
----UFO data----
Scraped from: http://www.nuforc.org/webreports.html
'''



ufo = pd.read_csv('https://raw.githubusercontent.com/sinanuozdemir/SF_DAT_15/master/data/ufo.csv')   


ufo.head()              # Look at the top 5 observations
ufo.tail()              # Look at the bottom 5 observations
ufo.describe()          # describe any numeric columns (unless all columns are non-numeric)
ufo.columns             # column names (which is "an index")



ufo['Location'] = ufo['City'] + ', ' + ufo['State']

ufo.head()


ufo.rename(columns={'Colors Reported':'Colors', 'Shape Reported':'Shape'}, inplace=True)


ufo.head()


del ufo['City']                  # delete a column (permanently)
del ufo['State']                 # delete a column (permanently)


ufo.Shape.value_counts()                # excludes missing values
ufo.Shape.value_counts(dropna=False)    # includes missing values


ufo.Shape.isnull().sum() # count the missing values in the shape column

ufo.isnull().sum()       # returns a count of missing values in all columns

# Shows how many rows has a not null shape AND a not null color
ufo[(ufo.Shape.notnull()) & (ufo.Colors.notnull())]



ufo.dropna()             # drop a row if ANY values are missing
ufo.dropna(how='all')    # drop a row only if ALL values are missing

ufo                      # Without an inplace=True, the dataframe is unaffected!

ufo.Colors.fillna(value='Unknown', inplace=True)

ufo.fillna(value = 'Unknown')   # Temporary

ufo

ufo.fillna(value = 'Unknown', inplace = True)   # Permanent


