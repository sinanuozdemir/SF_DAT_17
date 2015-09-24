
'''
Data Science in Python

    ----UFO data----
    Scraped from: http://www.nuforc.org/webreports.html
'''

'''
Reading, Summarizing data
'''

import pandas as pd

# Running this next line of code assumes that your console working directory is set up correctly 
# To set up your working directory
#        1) Put the data and the script in the same working directory
#        2) Select the options buttom in the upper right hand cornder of the editor
#        3) Select "Set console working directory"

ufo = pd.read_csv('https://raw.githubusercontent.com/sinanuozdemir/SF_DAT_17/master/data/ufo.csv')   # can also read csvs directly from the web!

ufo                 
ufo.head(5)          # Look at the top x observations
ufo.tail()            # Bottom x observations (defaults to 5)
ufo.describe()        # describe any numeric columns (unless all columns are non-numeric)
ufo.index             # "the index" (aka "the labels")
ufo.columns           # column names (which is "an index")
ufo.shape 		    # gives us a tuple of (# rows, # cols)

# DataFrame vs Series, selecting a column
type(ufo)
ufo['State']
ufo.State            # equivalent
ufo['Shape Reported']	# Must use the [''] notation if column name has a space
type(ufo.State)

# summarizing a non-numeric column
ufo.State.describe()        # Only works for non-numeric if you don't have any numeric data 
ufo.State.value_counts()    # Valuable if you have numeric columns, which you often will

# You can add in a sort optional arguement
ufo.State.value_counts(sort = True)


ufo.shape[0]                # number of rows

ufo.State.value_counts() / ufo.shape[0] # Values divided by number of records
# Shows percentages of sightings for each state

'''
Slicing / Filtering / Sorting
'''

ufo 					# Sanity check, nothing has changed!

# selecting multiple columns
ufo[['State', 'City','Shape Reported']]
my_cols = ['State', 'City']
ufo[my_cols]
type(ufo[my_cols])

'''
Notation

[row_start_index:row_end_index , col_start_index:col_end_index]
rows row_start_index through row_end_index and columns col_start_index through col_end_index
OR
[row_index , col_start_index:col_end_index]
only row row index and columns col_start_index through col_end_index
OR
[row_start_index:row_end_index , col_index]
rows row_start_index through row_end_index and only column col_index
OR
[[row1, row2], col_index]
only rows row1 and row2 and column col_index


'''

    
# logical filtering

ufo.State == 'TX'  # the == will compare 'TX' with every element in the column
    
# if we put the series of Trues and Falses in the dataframe, we will
# only get the rows where it is True, otherwise we won't see it!
    
ufo[ufo.State == 'TX']
# only TX sightings

# not TX sightings
ufo[~(ufo.State == 'TX')]   
ufo[(ufo.State == 'TX') == False]
ufo[(ufo.State != 'TX')]                # All the same!


ufo.City[ufo.State == 'TX']
# only cities of texas sightings
ufo[ufo.State == 'TX'].City             # Same thing


ufo[(ufo.State == 'CA') | (ufo.State =='TX')] # CA OR  TX
ufo[(ufo.State == 'CA') & (ufo.State =='TX')] # CA AND TX


ufo_dallas = ufo[(ufo.City == 'Dallas') & (ufo.State =='TX')]
ufo[ufo.City.isin(['Austin','Dallas', 'Houston'])]

# sorting
ufo.State.order()                               # only works for a Series
ufo.sort_index(by='State')                      # sort rows by specific column
ufo.sort_index(by=['State', 'Shape Reported'])  # sort by multiple columns
ufo.sort_index(by=['State', 'Shape Reported'], ascending=[False, True])  # specify sort order

ufo                                             # sort_index won't change the dataframe!

# unless we tell it to with inplace = True
ufo.sort_index(by='State', inplace = True)      # sort rows by specific column

ufo
# Now it's changed!

# detecting duplicate rows
ufo.duplicated()                                # Series of logicals
ufo.duplicated().sum()                          # count of duplicates
ufo[ufo.duplicated(['State','Time'])]           # only show duplicates
ufo[ufo.duplicated()==False]                    # only show unique rows
ufo_unique = ufo[~ufo.duplicated()]             # only show unique rows
ufo.duplicated(['State','Time']).sum()          # columns for identifying duplicates


''' EXERCISE '''


# from before: this gives us the percentage of sightings by state
ufo.State.value_counts() / ufo.shape[0] # Values divided by number of records

# this dataframe is only sightings in texas
ufo_texas = ufo[ufo.State == 'TX']

# Use value counts to display the 

# Answer below so no peeking!!

# BONUS, sort the dataframe so the city with the highest frequency is at the top


# Select the shape reported of all sightings in Connectucut








# ANSWER
ufo_texas.City.value_counts(sort = True) / ufo_texas.shape[0]

ufo['Shape Reported'][ufo.State=='CT']


'''
Modifying Columns
'''

# add a new column as a function of existing columns
ufo['Location'] = ufo['City'] + ', ' + ufo['State']
ufo.head()

# rename columns inplace
ufo.rename(columns={'Colors Reported':'Colors', 'Shape Reported':'Shape'}, inplace=True)

ufo.head()

# hide a column (temporarily)
ufo.drop(['Location'], axis=1)
# axis = 1 means column as opposed to axis = 0 (row)

ufo                 # not changed!

# delete a column (permanently)
del ufo['Location']

ufo                 # changed!

''' Exercise '''
# Make a new column, called 

'''
Handling Missing Values
'''

# missing values are often just excluded
ufo.describe()                          # excludes missing values
ufo.Shape.value_counts()                # excludes missing values
ufo.Shape.value_counts(dropna=False)    # includes missing values (new in pandas 0.14.1)

# find missing values in a Series
ufo.Shape.isnull()       # True if NaN, False otherwise
ufo.Shape.isnull().sum() # count the missing values

ufo.Shape.notnull()      # False if NaN, True otherwise


# Shows which rows do not have a shape designation
ufo[ufo.Shape.isnull()]
# Shows how many rows has a not null shape AND a not null color
ufo[(ufo.Shape.notnull()) & (ufo.Colors.notnull())]

# Makes a new dataframe with not null shape designations
ufo_shape_not_null = ufo[ufo.Shape.notnull()]


# drop missing values
ufo.dropna()             # drop a row if ANY values are missing
ufo.dropna(how='all')    # drop a row only if ALL values are missing

ufo                      # Remember, without an inplace=True, the dataframe is unaffected!

# fill in missing values to Colors only
ufo.Colors.fillna(value='Unknown', inplace=True)

# calling fillna on a dataframe will replace ALL null values
ufo.fillna('Unknown')                   # Temporary
ufo.fillna('Unknown', inplace = True)   # Permanent


ufo[ufo.Shape=='TRIANGLE'].shape[0]

ufo.Shape.replace('DELTA', 'TRIANGLE', inplace = True)   # replace values in a Series
ufo.replace('PYRAMID', 'TRIANGLE', inplace = True)       # replace values throughout a DataFrame

ufo[ufo.Shape=='TRIANGLE'].shape[0]




''' Fun Stuff '''

# Make a new month column
ufo['Month'] = ufo['Time'].apply(lambda x:int(x.split('/')[0]))

'''
the apply function applys the lambda funciton to every element in the Series

lambda x:x.split('/')[0] will take in x and split it by '/' and return the first element
so if we pass in say 9/3/2014 01:22 into the function we would get:

9                i.e. the month

'''

# similar for day
ufo['Day'] = ufo['Time'].apply(lambda x:int(x.split('/')[1]))

# for year, I need the [:4] at the end to remove the time
ufo['Year'] = ufo['Time'].apply(lambda x:int(x.split('/')[2][:4]))


# Plot of sightings per day in 2013
ufo[ufo.Year==2013].Day.value_counts().sort_index().plot()


# Plot the number of sightings over time
sightings_per_year = ufo.groupby('Year').City.count()

sightings_per_year.plot(kind='line', 
                        color='r', 
                        linewidth=1, 
                        title='UFO Sightings by year')
# -----Analysis-----
# Clearly, Aliens love the X-Files (which came out in 1993).
# Aliens are a natural extension of the target demographic so it makes sense.

# Well hold on Sinan, the US population is always increasing
# So maybe there's a jump in population which would make sense!
# US Population data from 1930 as taken from the Census

us_population = pd.read_csv('../data/us_population.csv')
us_population.plot(x = 'Date', y = 'Population', legend = False)

# Seems like a steady increase to me..


# Plot the sightings in in July 
ufo[(ufo.Year==2014) & (ufo.Month == 7)].groupby('Day').City.count().plot(  kind='bar',
                                                        color='b', 
                                                        title='UFO Sightings in July 2014')
                                                        


# -----Analysis-----
# Aliens are love the 4th of July. The White House is still standing. Therefore
# it follows that Aliens are just here for the party.

# Well maybe it's just 2014?

# Plot multiple plots on the same plot (plots neeed to be in column format)
ufo_fourth = ufo[(ufo.Year.isin([2011, 2012, 2013, 2014])) & (ufo.Month == 7)]
ufo_fourth.groupby(['Year', 'Day']).City.count().unstack(0).plot(   kind = 'bar', figsize=(7,9))


# unstack will take a groupby of multiple indices and split it by column (mainly great for sub plotting)

# Hmm let's make that prettier by making it 4 seperate charts
ufo_fourth.groupby(['Year', 'Day']).City.count().unstack(0).plot(
                                        kind = 'bar',
                                        subplots=True, 
                                        figsize = (7,9))


'''
Writing Data
'''

ufo.to_csv('ufo_new.csv')               # First column is an index
ufo.to_csv('ufo_new.csv', index=False)  # First column is no longer index


'''
The IRIS dataset. Taken from the Machine Learning Repository from UCI
https://archive.ics.uci.edu/ml/datasets/Iris

'''


# load the famous iris data
iris = pd.read_csv('https://raw.githubusercontent.com/sinanuozdemir/SF_DAT_17/master/data/iris.csv')   # can also read csvs directly from the web!
# Read data into pandas and explore



# explore data numerically, looking for differences between species
iris.describe()
iris.groupby('species').sepal_length.mean()
iris.groupby('species')['sepal_length', 'sepal_width', 'petal_length', 'petal_width'].mean()
iris.groupby('species').describe()

# explore data by sorting, looking for differences between species
iris.sort_index(by='sepal_length')
iris.sort_index(by='sepal_width')
iris.sort_index(by='petal_length')
iris.sort_index(by='petal_width')


# explore data visually, looking for differences between species
iris.petal_width.hist(by=iris.species, sharex=True)
iris.boxplot(column='petal_width', by='species')
iris.boxplot(by='species')

################
### EXERCISE ###
################

'''
create a function called color_flower that takes in a string

if the string inputed is "Iris-setosa":
    return "b"
else if the string inputted is "Iris-virginica":
    return "r"
else:
    return "g"


Solution is below so no peeking!

'''









def color_flower(flower_name):
    if flower_name == 'Iris-setosa':
        return 'b'
    elif flower_name == 'Iris-virginica':
        return 'r'
    else:
        return 'g'

# apply this function to the species column to give us 
# designated colors!

colors = iris.species.apply(color_flower)

colors


iris.plot(x='petal_length', y='petal_width', kind='scatter', c=colors)




pd.scatter_matrix(iris, c=colors, figsize = (10,10))
# look at petal length vs petal width


