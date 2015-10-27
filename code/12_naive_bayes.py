'''
CLASS: Naive Bayes SMS spam classifier using sklearn
Data source: https://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection
'''

## READING IN THE DATA

# read tab-separated file using pandas
import pandas as pd
df = pd.read_table('https://raw.githubusercontent.com/sinanuozdemir/SF_DAT_15/master/data/sms.tsv',
                   sep='\t', header=None, names=['label', 'msg'])

# examine the data
df.head(30)
df.label.value_counts()
df.msg.describe()

# convert label to a binary variable
df['label'] = df.label.map({'ham':0, 'spam':1})
df.head()

# split into training and testing sets
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df.msg, df.label, random_state=1)
X_train.shape
X_test.shape


## COUNTVECTORIZER: 'convert text into a matrix of token counts'
## http://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html

from sklearn.feature_extraction.text import \
CountVectorizer

# start with a simple example
train_simple = ['call you tonight',
                'Call me a cab',
                'please call me... PLEASE 44!']

# learn the 'vocabulary' of the training data
vect = CountVectorizer()
# fit learns the vocab
vect.fit(train_simple)
vect.get_feature_names()

# transform training data into a 'document-term matrix'
train_simple_dtm = vect.transform(train_simple)
train_simple_dtm
train_simple_dtm.toarray()

# examine the vocabulary and document-term matrix together
pd.DataFrame(train_simple_dtm.toarray(), \
columns=vect.get_feature_names())

# transform testing data into a document-term matrix (using existing vocabulary)
test_simple = ["please don't call me"]
test_simple_dtm = vect.transform(test_simple)
test_simple_dtm.toarray()
pd.DataFrame(test_simple_dtm.toarray(), columns=vect.get_feature_names())


## REPEAT PATTERN WITH SMS DATA

# instantiate the vectorizer
vect = CountVectorizer()

# learn vocabulary and create document-term matrix in a single step
train_dtm = vect.fit_transform(X_train)
train_dtm

# transform testing data into a document-term matrix
test_dtm = vect.transform(X_test)
test_dtm

# store feature names and examine them
train_features = vect.get_feature_names()
len(train_features)
train_features[:50]
train_features[-50:]

# convert train_dtm to a regular array
train_arr = train_dtm.toarray()
train_arr


## SIMPLE SUMMARIES OF THE TRAINING DATA

# refresher on numpy
import numpy as np
arr = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])

arr


arr[0, 0]
arr[1, 3]
arr[0, :]
arr[:, 0]
np.sum(arr)
np.sum(arr, axis=0)
np.sum(arr, axis=1)

# exercise: calculate the number of tokens in the
# 0th message in train_arr








X_train[0]
sum(train_arr[0, :])


# exercise: count how many times the 
# 0th token appears across ALL messages in train_arr







train_features[0]
sum(train_arr[:, 0])

# exercise: count how many times EACH token 
# appears across ALL messages in train_arr






np.sum(train_arr, axis=0)

# create a DataFrame of tokens with their counts
train_token_counts = pd.DataFrame({'token':train_features, 'count':np.sum(train_arr, axis=0)})

train_token_counts

train_token_counts.sort_index(by='count', ascending=False)

train_token_counts[train_token_counts.token=='00']




## MODEL BUILDING WITH NAIVE BAYES
## http://scikit-learn.org/stable/modules/naive_bayes.html

# train a Naive Bayes model using train_dtm
from sklearn.naive_bayes import MultinomialNB
nb = MultinomialNB()
nb.fit(train_dtm, y_train)

# make predictions on test data using test_dtm
preds = nb.predict(test_dtm)
preds

# compare predictions to true labels
from sklearn import metrics
print metrics.accuracy_score(y_test, preds)
print metrics.confusion_matrix(y_test, preds)

# predict (poorly calibrated) probabilities and calculate AUC
probs = nb.predict_proba(test_dtm)[:, 1]
probs
print metrics.roc_auc_score(y_test, probs)

# exercise: show the message text for the false positives







X_test[y_test < preds]

# exercise: show the message text for the false negatives





X_test[y_test > preds]


## COMPARE NAIVE BAYES AND LOGISTIC REGRESSION
## USING ALL DATA AND CROSS-VALIDATION

# create a document-term matrix using all data
all_dtm = vect.fit_transform(df.msg)

# instantiate logistic regression
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()

# compare AUC using cross-validation
from sklearn.cross_validation import cross_val_score
cross_val_score(nb, all_dtm, df.label, cv=10, scoring='roc_auc').mean()
cross_val_score(logreg, all_dtm, df.label, cv=10, scoring='roc_auc').mean()

#note the pure speed!!! nb was way faster




## EXERCISE adding in n_grams

# an n_gram is a n word phrase. So a 3 gram includes
# "I have a" or "are a winner"

vect = CountVectorizer(stop_words='english', ngram_range=[1,5]) 

# learn vocabulary and create document-term matrix in a single step
train_dtm = vect.fit_transform(X_train)
train_dtm

# transform testing data into a document-term matrix
test_dtm = vect.transform(X_test)
test_dtm

nb = MultinomialNB()
nb.fit(train_dtm, y_train)

# make predictions on test data using test_dtm
preds = nb.predict(test_dtm)
preds

# compare predictions to true labels
print metrics.accuracy_score(y_test, preds)
print metrics.confusion_matrix(y_test, preds)

# predict (poorly calibrated) probabilities and calculate AUC
probs = nb.predict_proba(test_dtm)[:, 1]
probs
print metrics.roc_auc_score(y_test, probs)

# show the message text for the false positives
X_test[y_test < preds]

# show the message text for the false negatives
X_test[y_test > preds]


## COMPARE NAIVE BAYES AND LOGISTIC REGRESSION
## USING ALL DATA AND CROSS-VALIDATION

# create a document-term matrix using all data
all_dtm = vect.fit_transform(df.msg)

all_dtm


# instantiate logistic regression
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()

# compare AUC using cross-validation
cross_val_score(nb, all_dtm, df.label, cv=10, scoring='roc_auc').mean()
cross_val_score(logreg, all_dtm, df.label, cv=10, scoring='roc_auc').mean()


# a lot slower!!!!



# EXERCISE try a naive bayes classification using
# n grams ranging from 1 to 10 and display each model's
# cross validated roc_auc score.
# hint you can write your own for loop or use gridsearch


# Graph the results with number of n grams used on the x axis
# and cross validated roc_auc as your y axis









## EXTRA  EXERCISE: CALCULATE THE 'SPAMMINESS' OF EACH TOKEN

# create separate DataFrames for ham and spam
df_ham = df[df.label==0]

df_ham


df_spam = df[df.label==1]

df_spam

# learn the vocabulary of ALL messages and save it

vect = CountVectorizer() 

vect.fit(df.msg)
all_features = vect.get_feature_names()

all_features

# create document-term matrix of ham, then convert to a regular array
ham_dtm = vect.transform(df_ham.msg)
ham_arr = ham_dtm.toarray()

ham_arr

# create document-term matrix of spam, then convert to a regular array
spam_dtm = vect.transform(df_spam.msg)
spam_arr = spam_dtm.toarray()

spam_arr

# count how many times EACH token appears across 
# ALL messages in ham_arr
ham_counts = np.sum(ham_arr, axis=0)

# count how many times EACH token appears across 
# ALL messages in spam_arr
spam_counts = np.sum(spam_arr, axis=0)

# create a DataFrame of tokens with their separate ham and spam counts
all_token_counts = pd.DataFrame({'token':all_features, 'ham':ham_counts, 'spam':spam_counts})

# add one to ham counts and spam counts so that ratio calculations (below) make more sense
all_token_counts['ham'] = all_token_counts.ham + 1
all_token_counts['spam'] = all_token_counts.spam + 1

# calculate ratio of spam-to-ham for each token
all_token_counts['spam_ratio'] = all_token_counts.spam / all_token_counts.ham
all_token_counts.sort_index(by='spam_ratio')
