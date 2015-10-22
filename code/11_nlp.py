'''
CLASS: Natural Language Processing

Adapted from: https://github.com/charlieg/A-Smattering-of-NLP-in-Python

What is NLP?
- Using computers to process (analyze, understand, generate) natural human languages

Why NLP?
- Most knowledge created by humans is unstructured text
- Need some way to make sense of it
- Enables quantitative analysis of text data

Why NLTK?
- High-quality, reusable NLP functionality
'''


# If you haven't done so, DO THIS NOW
import nltk
nltk.download()


'''
Tokenization

What:  Separate text into units such as sentences or words
Why:   Gives structure to previously unstructured text
Notes: Relatively easy with English language text, not easy with some languages
'''

# "corpus" = collection of documents
# "corpora" = plural form of corpus

import requests
from bs4 import BeautifulSoup

r = requests.get("http://en.wikipedia.org/wiki/Data_science")
b = BeautifulSoup(r.text)
paragraphs = b.find("body").findAll("p")
text = ""
for paragraph in paragraphs:
    text += paragraph.text + " "
# Data Science corpus
text[:500]

# tokenize into sentences
sentences = [sent for sent in nltk.sent_tokenize(text)]
sentences[:10]

# tokenize into words
tokens = [word for word in nltk.word_tokenize(text)]
tokens[:100]

# only keep tokens that start with a letter (using regular expressions)
import re
clean_tokens = [token for token in tokens if re.search('^[a-zA-Z]+', token)]
clean_tokens[:100]

# count the tokens
from collections import Counter
c = Counter(clean_tokens)

c.most_common(25)       # mixed case

sorted(c.items())[:25]  # counts similar words separately
for item in sorted(c.items())[:25]:
    print item[0], item[1]

###################
##### EXERCISE ####
###################

# Put each word in clean_tokens in lower case
# find the new word count of the lowered tokens
# Then show the top 10 words used in this corpus














# ANSWER
clean_tokens_lowered = [ctok.lower() for ctok in clean_tokens]
c = Counter(clean_tokens_lowered)
c.most_common(10)       # mixed case



'''
Stemming
What:  Reduce a word to its base/stem form
Why:   Often makes sense to treat multiple word forms the same way
Notes: Uses a "simple" and fast rule-based approach
       Output can be undesirable for irregular words
       Stemmed words are usually not shown to users (used for analysis/indexing)
       Some search engines treat words with the same stem as synonyms
'''

from nltk.stem.snowball import SnowballStemmer
stemmer = SnowballStemmer('english')

# example stemming
stemmer.stem('charge')
stemmer.stem('charging')
stemmer.stem('charged')

# stem the tokens
stemmed_tokens = [stemmer.stem(t) for t in clean_tokens]

# count the stemmed tokens
c = Counter(stemmed_tokens)
c.most_common(25)       # all lowercase
sorted(c.items())[:25]  # some are strange


'''
Lemmatization
What:  Derive the canonical form ('lemma') of a word
Why:   Can be better than stemming, reduces words to a 'normal' form.
Notes: Uses a dictionary-based approach (slower than stemming)
'''

lemmatizer = nltk.WordNetLemmatizer()

# compare stemmer to lemmatizer
stemmer.stem('dogs')
lemmatizer.lemmatize('dogs')

stemmer.stem('wolves') # Beter for information retrieval and search
lemmatizer.lemmatize('wolves') # Better for text analysis

stemmer.stem('is')
lemmatizer.lemmatize('is')
lemmatizer.lemmatize('is',pos='v')

'''
Part of Speech Tagging
What:  Determine the part of speech of a word
Why:   This can inform other methods and models such as Named Entity Recognition
Notes: http://www.ling.upenn.edu/courses/Fall_2003/ling001/penn_treebank_pos.html
'''

temp_sent = 'Sinan and Patrick and Liam are great teachers!'
# pos_tag takes a tokenize sentence
nltk.pos_tag(nltk.word_tokenize(temp_sent))


'''
Stopword Removal
What:  Remove common words that will likely appear in any text
Why:   They don't tell you much about your text
'''

# most of top 25 stemmed tokens are "worthless"
c.most_common(25)

# view the list of stopwords
stopwords = nltk.corpus.stopwords.words('english')
sorted(stopwords)


##################
### Exercise  ####
##################


# Create a variable called stemmed_stops which is the 
# stemmed version of each stopword in stopwords
# Use the stemmer we used up above!

# Then create a list called stemmed_tokens_no_stop that 
# contains only the tokens in stemmed_tokens that aren't in 
# stemmed_stops

# Show the 25 most common stemmed non stop word tokens










# Answers

# stem the stopwords
stemmed_stops = [stemmer.stem(t) for t in stopwords]

# remove stopwords from stemmed tokens
stemmed_tokens_no_stop = [t for t in stemmed_tokens if t not in stemmed_stops]
c = Counter(stemmed_tokens_no_stop)

#25 most common tokens
c.most_common(25)



# remove stopwords from cleaned tokens
clean_tokens_no_stop = [t for t in clean_tokens if t not in stopwords]
c = Counter(clean_tokens_no_stop)
most_common_not_stemmed = c.most_common(25)




'''
Named Entity Recognition
What:  Automatically extract the names of people, places, organizations, etc.
Why:   Can help you to identify "important" words
Notes: Training NER classifier requires a lot of annotated training data
       Should be trained on data relevant to your task
       Stanford NER classifier is the "gold standard"
'''

sentence = 'Sinan is an instructor for General Assembly'

tokenized = nltk.word_tokenize(sentence)

tokenized

tagged = nltk.pos_tag(tokenized)

tagged

chunks = nltk.ne_chunk(tagged)

chunks

# Note how chunks put general assembly as ONE entity!

def extract_entities(text):
    entities = []
    # tokenize into sentences
    for sentence in nltk.sent_tokenize(text):
        # tokenize sentences into words
        # add part-of-speech tags
        # use NLTK's NER classifier
        chunks = nltk.ne_chunk(nltk.pos_tag(nltk.word_tokenize(sentence)))
        # parse the results
        entities.extend([chunk for chunk in chunks if hasattr(chunk, 'label')])
    return entities

for entity in extract_entities('Sinan is an instructor for General Assembly'):
    print '[' + entity.label() + '] ' + ' '.join(c[0] for c in entity.leaves())


'''
Term Frequency - Inverse Document Frequency (TF-IDF)
What:  Computes "relative frequency" that a word appears in a document
           compared to its frequency across all documents
Why:   More useful than "term frequency" for identifying "important" words in
           each document (high frequency in that document, low frequency in
           other documents)
Notes: Used for search engine scoring, text summarization, document clustering
How: 
    TF(t) = (Number of times term t appears in a document) / (Total number of terms in the document).
    IDF(t) = log_e(Total number of documents / Number of documents with term t in it).
'''

sample = ['Bob likes sports', 'Bob hates sports', 'Bob likes likes trees']

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

vect = CountVectorizer()

# Each row represents a sentence
# Each column represents a word
vect.fit_transform(sample).toarray()
vect.get_feature_names()


from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer()
tfidf.fit_transform(sample).toarray()
tfidf.get_feature_names()

# the IDF of each word
idf = tfidf.idf_
print dict(zip(tfidf.get_feature_names(), idf))


###############
## Exercise ###
###############


# for each sentence in sample, find the most "interesting 
#words" by ordering their tfidf in ascending order








# Answer


for index, sentence in enumerate(sample): #enumerate gives you the index of your for loop
    zipped_words_tfidf = zip(tfidf.get_feature_names(), tfidf.fit_transform(sample).toarray()[index])
    # zip together the words and their tfidf score
    
    print sentence
    print sorted(zipped_words_tfidf, key=lambda x: x[1])[-2:]
    # print the last 2 most "interesting words"
    




###############
## Exercise ###
###############

sentences

# make a TFIDF for the sentences from the wiki article sentences











# Answer

tfidf = TfidfVectorizer()
tfidf.fit_transform(sentences)
# sparse matrix

saved_array = tfidf.fit_transform(sentences).toarray()
tfidf.get_feature_names()

for index, sentence in enumerate(sentences): #enumerate gives you the index of your for loop
    zipped_words_tfidf = zip(tfidf.get_feature_names(), saved_array[index])
    # zip together the words and their tfidf score
    print sentences[index], sorted(zipped_words_tfidf, key=lambda x: x[1])[-2:]
    # print the last 2 most "interesting words"
    


'''
LDA - Latent Dirichlet Allocation
What:  Way of automatically discovering topics from sentences
Why:   Much quicker than manually creating and identifying topic clusters
'''
import lda

# Instantiate a count vectorizer with two additional parameters
vect = CountVectorizer(stop_words='english', ngram_range=[1,3]) 
sentences_train = vect.fit_transform(sentences)

# Instantiate an LDA model
model = lda.LDA(n_topics=10, n_iter=500)
model.fit(sentences_train) # Fit the model 
n_top_words = 10
topic_word = model.topic_word_
for i, topic_dist in enumerate(topic_word):
    topic_words = np.array(vect.get_feature_names())[np.argsort(topic_dist)][:-n_top_words:-1]
    print('Topic {}: {}'.format(i, ', '.join(topic_words)))


'''
EXAMPLE: Automatically summarize a document
'''

# corpus of 2000 movie reviews
from nltk.corpus import movie_reviews
reviews = [movie_reviews.raw(filename) for filename in movie_reviews.fileids()]

# create document-term matrix
tfidf = TfidfVectorizer(stop_words='english')
dtm = tfidf.fit_transform(reviews)
features = tfidf.get_feature_names()

import numpy as np

# find the most and least "interesting" sentences in a randomly selected review
def summarize():
    
    # choose a random movie review    
    review_id = np.random.randint(0, len(reviews))
    review_text = reviews[review_id]

    # we are going to score each sentence in the review for "interesting-ness"
    sent_scores = []
    # tokenize document into sentences
    for sentence in nltk.sent_tokenize(review_text):
        # exclude short sentences
        if len(sentence) > 6:
            score = 0
            token_count = 0
            # tokenize sentence into words
            tokens = nltk.word_tokenize(sentence)
            # compute sentence "score" by summing TFIDF for each word
            for token in tokens:
                if token in features:
                    score += dtm[review_id, features.index(token)]
                    token_count += 1
            # divide score by number of tokens
            sent_scores.append((score / float(token_count + 1), sentence))

    # lowest scoring sentences
    print '\nLOWEST:\n'
    for sent_score in sorted(sent_scores)[:3]:
        print sent_score[1]

    # highest scoring sentences
    print '\nHIGHEST:\n'
    for sent_score in sorted(sent_scores, reverse=True)[:3]:
        print sent_score[1]

# try it out!
summarize()


'''
TextBlob Demo: "Simplified Text Processing"
Installation: pip install textblob
'''

from textblob import TextBlob, Word

# identify words and noun phrases
blob = TextBlob('Liam and Sinan are instructors for General Assembly')
blob.words
blob.noun_phrases

# sentiment analysis
blob = TextBlob('I hate this horrible movie. This movie is not very good.')
blob.sentences
blob.sentiment.polarity
[sent.sentiment.polarity for sent in blob.sentences]

# sentiment subjectivity
TextBlob("I am a cool person").sentiment.subjectivity # Pretty subjective
TextBlob("I am a person").sentiment.subjectivity # Pretty objective
# different scores for essentially the same sentence
print TextBlob('Liam and Sinan are instructors for General Assembly in San Francisco').sentiment.subjectivity
print TextBlob('Patrick and Sinan are instructors in Texas').sentiment.subjectivity

# singularize and pluralize
blob = TextBlob('Put away the dishes.')
[word.singularize() for word in blob.words]
[word.pluralize() for word in blob.words]

# spelling correction
blob = TextBlob('15 minuets late')
blob.correct()

# spellcheck
Word('parot').spellcheck()

# definitions
Word('bank').define()
Word('bank').define('v')

# translation and language identification
blob = TextBlob('Welcome to the classroom.')
blob.translate(to='es')
blob = TextBlob('Hola amigos')
blob.detect_language()

