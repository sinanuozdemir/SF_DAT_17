## Class 12 LabHomework: Yelp Reviews

This assignment uses a small subset of the data from Kaggle's [Yelp Business Rating Prediction](https://www.kaggle.com/c/yelp-recsys-2013) competition.

**Description of the data:**

`yelp.csv` (in the data folder) contains the Yelp ratings data
* Each observation in this dataset is a review of a particular business by a particular user.
* The "stars" column is the number of stars (1 through 5) assigned by the reviewer to the business. (Higher stars is better.)
* The "cool" column is the number of "cool" votes this particular review received from other Yelp users. There is no limit to how many "cool" votes a review can receive.
* The "useful" and "funny" columns are similar to the "cool" column.

**tasks**

1. Use Count and Tfidf Vectorizer to turn the text data into numerical columns.
2. Experiment with parameters, "max_features", and "ngram_range"
3. Use these new features, along with your old features to see if you can predict stars with more ease!
4. Use Logistic Regression and KNN as well to compare models. (Hint, when comparing across classification models, ROC/AUC is a useful metric) and try to get the best metrics!