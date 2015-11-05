import requests
from BeautifulSoup import BeautifulSoup
from sklearn.feature_extraction.text import CountVectorizer

# grab postings from the web
texts = []
for i in range(0,1000,10): # cycle through 100 pages of indeed job resources
    soup = BeautifulSoup(requests.get('http://www.indeed.com/jobs?q=data+scientist&start='+str(i)).text)
    texts += [a.text for a in soup.findAll('span', {'class':'summary'})]


    

print len(texts), "job descriptions" # 1,500 descriptions


texts[0]   # first job description


# I will encode each string in ascii
texts = [t.encode('ascii', 'ignore') for t in texts]


texts[0]   # encoded in ascii




#############################
## Actual Machine Learning ##
#############################


vect = CountVectorizer(ngram_range=(1,1), stop_words='english')
# make a count vectorizer to get basic counts

matrix = vect.fit_transform(texts)
# fit and learn to the vocabulary in the corpus

len(vect.get_feature_names())  # how many features are there


freqs = [(word, matrix.getcol(idx).sum()) for word, idx in vect.vocabulary_.items()]
#sort from largest to smallest
for phrase, times in sorted (freqs, key = lambda x: -x[1])[:25]:
    print phrase, times
    
    
    
# Tune a parameter    
    
# Try with up to 3 word phrases!
vect = CountVectorizer(ngram_range=(1,3), stop_words='english')
matrix = vect.fit_transform(texts)
len(vect.get_feature_names())  # how many features are there

freqs = [(word, matrix.getcol(idx).sum()) for word, idx in vect.vocabulary_.items()]
#sort from largest to smallest
for phrase, times in sorted (freqs, key = lambda x: -x[1])[:25]:
    print phrase, times