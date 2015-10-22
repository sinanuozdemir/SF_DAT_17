# read in tweets data into a dataframe
import sys
reload(sys)
sys.setdefaultencoding('utf8')

import pandas as pd
tweets = pd.read_csv('../data/so_many_tweets.csv')


# use the textblob module to get the sentiment of each
# tweet

from textblob import TextBlob

def stringToSentiment(string):
    return TextBlob(string).sentiment.polarity
    
stringToSentiment('i hate you')

tweets['sentiment'] = tweets.Text.map(stringToSentiment)

# Make a column called day which holds the unique
# day it was tweeted, e.g. 5/24/2015


tweets['day'] = tweets.Date.map(lambda x: x[:10])

sent = tweets.groupby('day')['sentiment'].mean()
volume = tweets.groupby('day')['Status'].mean()


sent.plot()

volume.plot()



# For each day, show the number of tweets and
# the average sentiment


# Show a graph of how volume "number of tweets"
# per day changed over the course of May



# Show a graph of how sentiment of tweets
# changed per day over the course of May


# Try taking out noise from the text like the 
# RT, any mentions or links and try again!