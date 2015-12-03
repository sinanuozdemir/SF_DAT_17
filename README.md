## SF DAT 17 Course Repository

Course materials for [General Assembly's Data Science course](https://generalassemb.ly/education/data-science/san-francisco/) in San Francisco, CA (9/14/15 - 12/02/15).

**Instructors:** Sinan Ozdemir (who is super cool!!!!!!)

**Teaching Assistants:**
David, Matt, and Sri (who are all way more awesome)

**Office hours:** All will be held in the student center at GA, 225 Bush Street


**[Course Project Information](project.md)**

**[Course Project Examples](project-examples.md)**


Monday | Wednesday
--- | ---
9/14: Introduction / Expectations / Intro to Data Science | 9/16: Git / Python 
9/21: Data Science Workflow / Pandas | 9/23: More Pandas!
9/28: Intro to Machine Learning / Numpy / KNN | 9/30: Scikit-learn / Model Evaluation<br>**Project Milestone:** Question and Data Set<br> **HW** Homework 1 Due
10/5: Linear Regression | 10/7: Logistic Regression
10/12: Columbus Day (NO CLASS) | 10/14: Working on a Data Problem
10/19: Clustering | 10/21: Natural Language Processing
10/26: Naive Bayes <br>**Milestone:** First Draft Due | 10/28: Decision Trees 
11/2: Ensembling Techniques  | 11/4: Dimension Reduction<br>**Milestone:** Peer Review Due
11/9 Support Vector Machines | 11/11: Web Development with Flask
11/16: Recommendation Engines | 11/18: Neural Networks Continued
11/23: SQL | 11/25: Turkey Day (NO CLASS)
11/30: Projects | 12/2: Projects


### Installation and Setup
* Install the [Anaconda distribution](http://continuum.io/downloads) of Python 2.7x.
* Install [Git](http://git-scm.com/book/en/v2/Getting-Started-Installing-Git) and create a [GitHub](https://github.com/) account.
* Once you receive an email invitation from [Slack](https://slack.com/), join our "SF\_DAT\_17 team" and add your photo!

### Resources
* [PEP 8 - Style Guide for Python](http://www.python.org/dev/peps/pep-0008)

### Class 1: Introduction / Expectations / Intro to Data Science
* Introduction to General Assembly
* Course overview: our philosophy and expectations ([slides](slides/01_course_overview.pdf))
* Intro to Data Science: ([slides](slides/01_intro_to_ds.pdf))
* Tools: check for proper setup of Git, Anaconda, overview of Slack

####Homework
* Make sure you have everything installed as specified above in "Installation and Setup" by Wednesday


### Class 2: Git / Python
* Introduction to [Git](slides/02_git_github.pdf)
* Intro to Python: ([code](code/02_python_refresher.py))

####Homework
* Go through the python file and finish any exercise you weren't able to in class
* Make sure you have all of the repos cloned and ready to go
	* You should have both "SF___DAT___17" and "SF___DAT___17__WORK"
* Read Greg Reda's [Intro to Pandas](http://www.gregreda.com/2013/10/26/intro-to-pandas-data-structures/) 

#### Resources:
* In depth Git/Github tutorial series made by a GA_DC  Data Science Instructor [here](https://www.youtube.com/playlist?list=PL5-da3qGB5IBLMp7LtN8Nc3Efd4hJq0kD)
* [Another Intro to Pandas](http://nbviewer.ipython.org/gist/wesm/4757075/PandasTour.ipynb) (Written by Wes McKinney and is adapted from his book)
	* [Here](https://vimeo.com/59324550) is a video of Wes McKinney going through his notebook!
	
	
	
	### Class 3: Pandas

**Agenda**

* Intro to Pandas walkthrough [here](code/03_pandas.py)
	* I will give you semi-cleaned data allowing us to work on step 3 of the data science workflow
	* Pandas is an excellent tool for exploratory data analysis
	* It allows us to easily manipulate, graph, and visualize basic statistics and elements of our data
	* [Pandas Lab!](code/03_pandas_lab.py)


**Homework**

* Begin thinking about potential projects that you'd want to work on. Consider the problems discussed in class today (we will see more next time and next Monday as well)
	* Do you want a predictive model?
	* Do you want to cluster similar objects (like words or other)?

**Resources:**

* Pandas
	 * [Split-Apply-Combine](http://i.imgur.com/yjNkiwL.png) pattern
    * Simple examples of [joins in Pandas](http://www.gregreda.com/2013/10/26/working-with-pandas-dataframes/#joining)
    * Check out this excellent example of [data wrangling and exploration in Pandas](http://nbviewer.ipython.org/github/cs109/content/blob/master/lec_04_wrangling.ipynb)
	    * For an extra challenge, try copying over the code into your own .py file
	* To learn more Pandas, review this [three-part tutorial](http://www.gregreda.com/2013/10/26/intro-to-pandas-data-structures/)
    * For more on Pandas plotting, read the [visualization page](http://pandas.pydata.org/pandas-docs/stable/visualization.html) from the official Pandas documentation.

    
    
### Class 4 - More Pandas

#### Agenda
* Class code on Pandas [here](code/04_more_pandas.py)
* We will work with 3 different data sets today:
	* the UFO dataset (as scraped from the [reporting website](http://www.nuforc.org/webreports.html)	
	* Fisher's Iris Dataset (as cleaned from a [machine learning repository](https://archive.ics.uci.edu/ml/datasets/Iris)
	* A dataset of (nearly) every FIFA goal ever scored (as scraped from the website)
* Pandas Lab! [here](code/04_more_pandas_lab.py)
	
	
####Homework
* Please review the [readme](hw/HW1-README.md) for the first homework. It is due NEXT Wednesday (9/30/2015)
* The one-pager for your project is also due. Please see [project guidelines](project.md)

### Class 5 - Intro to ML / Numpy / KNN

####Agenda

* Intro to numpy [code](code/05_numpy.py)
	* Numerical Python, code adapted from tutorial [here](http://www.engr.ucsb.edu/~shell/che210d/numpy.pdf)
	* Special attention to the idea of the np.array
* Intro to Machine Learning and KNN [slides](slides/05_ml_knn.pdf)
	* Supervised vs Unsupervised Learning
	* Regression vs. Classification
* Iris pre-work [code](code/05_iris_prework.py) and [code solutions](code/05_iris_prework_solutions.py)
	* Using numpy to investigate the iris dataset further
	* Understanding how humans learn so that we can teach the machine!
* [Lab](code/05_knn.py) to create our own KNN model


####Homework
* The one page project milestone as well as the pandas homework!
* Read this excellent article, [Understanding the Bias-Variance Tradeoff](http://scott.fortmann-roe.com/docs/BiasVariance.html), and be prepared to discuss it in class on Wednesday. (You can ignore sections 4.2 and 4.3.) Here are some questions to think about while you read:
    * In the Party Registration example, what are the features? What is the response? Is this a regression or classification problem?
    * In the interactive visualization, try using different values for K across different sets of training data. What value of K do you think is "best"? How do you define "best"?
    * In the visualization, what do the lighter colors versus the darker colors mean? How is the darkness calculated?
    * How does the choice of K affect model bias? How about variance?
    * As you experiment with K and generate new training data, how can you "see" high versus low variance? How can you "see" high versus low bias?
    * Why should we care about variance at all? Shouldn't we just minimize bias and ignore variance?
    * Does a high value for K cause over-fitting or under-fitting?

    
**Resources:**

* For a more in-depth look at machine learning, read section 2.1 (14 pages) of Hastie and Tibshirani's excellent book, [An Introduction to Statistical Learning](http://www-bcf.usc.edu/~gareth/ISL/). (It's a free PDF download!)


### Class 6: scikit-learn, Model Evaluation Procedures
* Introduction to scikit-learn with iris data ([code](code/06_sklearn_knn.py))
* Exploring the scikit-learn documentation: [user guide](http://scikit-learn.org/stable/modules/neighbors.html), [module reference](http://scikit-learn.org/stable/modules/classes.html#module-sklearn.neighbors), [class documentation](http://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html)
* Discuss the [article](http://scott.fortmann-roe.com/docs/BiasVariance.html) on the bias-variance tradeoff
* Look as some [code](code/06_bias_variance.py) on the bias variace tradeoff
	* To run this, I use a module called "seaborn" 
	* To install to anywhere in your terminal (git bash) and type in `sudo pip install seaborn`
* Model evaluation procedures ([slides](slides/06_model_evaluation_procedures.pdf), [code](code/06_model_evaluation_procedures.py))

**Homework:**

* Keep working on your project. Your [data exploration and analysis plan](project.md) is due in three weeks!

**Optional:**

* Practice what we learned in class today!
    * If you have gathered your project data already: Try using KNN for classification, and then evaluate your model. Don't worry about using all of your features, just focus on getting the end-to-end process working in scikit-learn. (Even if your project is regression instead of classification, you can easily convert a regression problem into a classification problem by converting numerical ranges into categories.)
    * If you don't yet have your project data: Pick a suitable dataset from the [UCI Machine Learning Repository](http://archive.ics.uci.edu/ml/datasets.html), try using KNN for classification, and evaluate your model. The [Glass Identification Data Set](http://archive.ics.uci.edu/ml/datasets/Glass+Identification) is a good one to start with.
    * Either way, you can submit your commented code to your SF_DAT_15_WORK, and we'll give you feedback.

**Resources:**

* Here's a great [30-second explanation of overfitting](http://www.quora.com/What-is-an-intuitive-explanation-of-overfitting/answer/Jessica-Su).
* For more on today's topics, these videos from Hastie and Tibshirani are useful: [overfitting and train/test split](https://www.youtube.com/watch?v=_2ij6eaaSl0) (14 minutes), [cross-validation](https://www.youtube.com/watch?v=nZAM5OXrktY) (14 minutes). (Note that they use the terminology "validation set" instead of "test set".)
    * Alternatively, read section 5.1 (12 pages) of [An Introduction to Statistical Learning](http://www-bcf.usc.edu/~gareth/ISL/), which covers the same content as the videos.
* This video from Caltech's machine learning course presents an [excellent, simple example of the bias-variance tradeoff](http://work.caltech.edu/library/081.html) (15 minutes) that may help you to visualize bias and variance.


### Class 7: Linear Regression
* Linear regression ([notebook](http://nbviewer.ipython.org/github/sinanuozdemir/SF_DAT_15/blob/master/code/07_linear_regression.ipynb), [notebook code](code/07_linear_regression.py))


* Yelp Lab [here](labs/07_yelp_reviews.md) with the [Yelp reviews data](data/yelp.csv). It is not required but your next homework will involve this dataset so it would be helpful to take a look now!

**Homework:**

* Watch these videos on [probability](https://www.youtube.com/watch?v=o4QmoNfW3bI) and [odds](https://www.youtube.com/watch?v=GxbXQjX7fC0) (8 minutes) if you're not familiar with either of those terms.
* Read these excellent articles from BetterExplained: [An Intuitive Guide To Exponential Functions & e](http://betterexplained.com/articles/an-intuitive-guide-to-exponential-functions-e/) and [Demystifying the Natural Logarithm (ln)](http://betterexplained.com/articles/demystifying-the-natural-logarithm-ln/).

**Resources:**

* Setosa has an excellent [interactive visualization](http://setosa.io/ev/ordinary-least-squares-regression/) of linear regression.
* To go much more in-depth on linear regression, read Chapter 3 of [An Introduction to Statistical Learning](http://www-bcf.usc.edu/~gareth/ISL/), from which this lesson was adapted. Alternatively, watch the [related videos](http://www.dataschool.io/15-hours-of-expert-machine-learning-videos/) or read my [quick reference guide](http://www.dataschool.io/applying-and-interpreting-linear-regression/) to the key points in that chapter.
* To learn more about Statsmodels and how to interpret the output, DataRobot has some decent posts on [simple linear regression](http://www.datarobot.com/blog/ordinary-least-squares-in-python/) and [multiple linear regression](http://www.datarobot.com/blog/multiple-regression-using-statsmodels/).
* This [introduction to linear regression](http://people.duke.edu/~rnau/regintro.htm) is much more detailed and mathematically thorough, and includes lots of good advice.
* This is a relatively quick post on the [assumptions of linear regression](http://pareonline.net/getvn.asp?n=2&v=8).
* John Rauser's talk on [Statistics Without the Agonizing Pain](https://www.youtube.com/watch?v=5Dnw46eC-0o) (12 minutes) gives a great explanation of how the null hypothesis is rejected.
* A major scientific journal recently banned the use of p-values:
    * Scientific American has a nice [summary](http://www.scientificamerican.com/article/scientists-perturbed-by-loss-of-stat-tools-to-sift-research-fudge-from-fact/) of the ban.
    * This [response](http://www.nature.com/news/statistics-p-values-are-just-the-tip-of-the-iceberg-1.17412) to the ban in Nature argues that "decisions that are made earlier in data analysis have a much greater impact on results".
    * Andrew Gelman has a readable [paper](http://www.stat.columbia.edu/~gelman/research/unpublished/p_hacking.pdf) in which he argues that "it's easy to find a p < .05 comparison even if nothing is going on, if you look hard enough".
* An article on ["P Hacking"](http://www.dailydot.com/geek/data-manipulation-tool-science-p-hacking/) the idea that you can alter data in order to achieve good p values

### Class 8: Logistic Regression
* Logistic regression ([notebook](http://nbviewer.ipython.org/github/sinanuozdemir/SF_DAT_17/blob/master/notebooks/08_logistic_regression.ipynb), [notebook code](code/08_logistic_regression_nb.py))
* Confusion matrix ([slides](slides/08_confusion_matrix.pdf))
* Exercise with Titanic data ([instructions](labs/08_titanic.md), [solution](labs/08_titanic_solutions.py))

**Homework:**

* If you aren't yet comfortable with all of the confusion matrix terminology, watch Rahul Patwari's videos on [Intuitive sensitivity and specificity](https://www.youtube.com/watch?v=U4_3fditnWg) (9 minutes) and [The tradeoff between sensitivity and specificity](https://www.youtube.com/watch?v=vtYDyGGeQyo) (13 minutes).

**Resources:**

* To go deeper into logistic regression, read the first three sections of Chapter 4 of [An Introduction to Statistical Learning](http://www-bcf.usc.edu/~gareth/ISL/), or watch the [first three videos](http://www.dataschool.io/15-hours-of-expert-machine-learning-videos/) (30 minutes) from that chapter.
* For a math-ier explanation of logistic regression, watch the first seven videos (71 minutes) from week 3 of Andrew Ng's [machine learning course](https://www.coursera.org/learn/machine-learning/home/info), or read the [related lecture notes](http://www.holehouse.org/mlclass/06_Logistic_Regression.html) compiled by a student.
* For more on interpreting logistic regression coefficients, read this excellent [guide](http://www.ats.ucla.edu/stat/mult_pkg/faq/general/odds_ratio.htm) by UCLA's IDRE and these [lecture notes](http://www.unm.edu/~schrader/biostat/bio2/Spr06/lec11.pdf) from the University of New Mexico.
* The scikit-learn documentation has a nice [explanation](http://scikit-learn.org/stable/modules/calibration.html) of what it means for a predicted probability to be calibrated.
* [Supervised learning superstitions cheat sheet](http://ryancompton.net/assets/ml_cheat_sheet/supervised_learning.html) is a very nice comparison of four classifiers we cover in the course (logistic regression, decision trees, KNN, Naive Bayes) and one classifier we do not cover (Support Vector Machines).
* This [simple guide to confusion matrix terminology](http://www.dataschool.io/simple-guide-to-confusion-matrix-terminology/) may be useful to you as a reference.



### Class 9: Working on a Data Problem
* Today we will work on a real world data problem! We will have 3 options.

* Option 1: (stocks) Use stock data from over 7 months of a fictional company ZYX including twitter sentiment, volume and stock price. Our goal is to create a predictive model that predicts forward returns. [data](data/ZYX_prices.csv) here
	* Project overview ([slides](slides/09_GA_Stocks.pdf))
	    * Be sure to read documentation thoroughly and ask questions! We may not have included all of the information you need...
* Option 2: Using ingredients to predict the type of recipe (Kaggle)[https://www.kaggle.com/c/whats-cooking]
* Option 3: San Francisco Crime Classification (Kaggle)[https://www.kaggle.com/c/sf-crime]




### Class 10: Clustering and Visualization
* The [slides](slides/10_clustering.pdf) today will focus on our first look at unsupervised learning, K-Means Clustering!
* The [code](code/10_kmeans_class_exercise.py) for today focuses on two main examples:
    * We will investigate simple clustering using the iris data set.
    * We will take a look at a harder example, using Pandora songs as data. See [data](data/songs.csv). See code [here](code/10_analyze_pandora.py)
    * Checking out some of the limitations of K-Means Clutering [here](code/10_kmeans_limitations.py)

**Homework:**

* **Project Milestone 2 is due in one week!**
* Download all of the NLTK collections.
   * In Python, use the following commands to bring up the download menu.
   * ```import nltk```
   * ```nltk.download()```
   * Choose "all".
   * Alternatively, just type ```nltk.download('all')```
* Install two new packages:  ```textblob``` and ```lda```.
   * Open a terminal or command prompt.
   * Type ```pip install textblob``` and ```pip install lda```.
   * 

**Resources:**

* [Introduction to Data Mining](http://www-users.cs.umn.edu/~kumar/dmbook/index.php) has a nice [chapter on cluster analysis](http://www-users.cs.umn.edu/~kumar/dmbook/ch8.pdf).
* The scikit-learn user guide has a nice [section on clustering](http://scikit-learn.org/stable/modules/clustering.html).




##Class 11: Natural Language Processing

**Agenda**

* Naural Language Processing is the science of turning words and sentences into data and numbers. Today we will be exploring techniques into this field
* [code](code/11_nlp.py) showing topics in NLP
* [lab](code/11_nlp_lab.py) analyzing tweets about the stock market


**Homework:**

* Read Paul Graham's [A Plan for Spam](http://www.paulgraham.com/spam.html) and be prepared to **discuss it in class on Wednesday**. Here are some questions to think about while you read:
    * Should a spam filter optimize for sensitivity or specificity, in Paul's opinion?
    * Before he tried the "statistical approach" to spam filtering, what was his approach?
    * How exactly does his statistical filtering system work?
    * What did Paul say were some of the benefits of the statistical approach?
    * How good was his prediction of the "spam of the future"?
* Below are the foundational topics upon which Wednesday's class will depend. Please review these materials before class:
    * **Confusion matrix:** [a good guide](http://www.dataschool.io/simple-guide-to-confusion-matrix-terminology/) roughly mirrors the lecture from class 10.
    * **Sensitivity and specificity:** Rahul Patwari has an [excellent video](https://www.youtube.com/watch?v=U4_3fditnWg&list=PL41ckbAGB5S2PavLIXUETzAmi5reIod23) (9 minutes).
    * **Basics of probability:** These [introductory slides](https://docs.google.com/presentation/d/1cM2dVbJgTWMkHoVNmYlB9df6P2H8BrjaqAcZTaLe9dA/edit#slide=id.gfc3caad2_00) (from the [OpenIntro Statistics textbook](https://www.openintro.org/stat/textbook.php)) are quite good and include integrated quizzes. Pay specific attention to these terms: probability, sample space, mutually exclusive, independent.
* You should definitely be working on your project! First draft is due Monday!!




##Class 12: Naive Bayes Classifier

Today we are going over advanced metrics for classification models and learning a brand new classification model called naive bayes!

**Agenda**

* Learn about ROC/AUC curves 
	* 	Slides [here](slides/12_classification_model_evaluation_metrics.pdf)
	* 	Code [here](code/12_confusion_roc.py)
* Learn the Naive Bayes Classifier 
	* 	Slides [here](slides/12_naive_bayes.pdf)
	* 	Code [here](code/12_naive_bayes.py)
	* In the code file above we will create our own spam classifier!


**Resources**

* Video on [ROC Curves](https://www.youtube.com/watch?v=21Igj5Pr6u4&list=PL41ckbAGB5S2PavLIXUETzAmi5reIod23) (12 minutes).
* My buddy's [blog post about the ROC video](http://www.dataschool.io/roc-curves-and-auc-explained/) includes the complete transcript and screenshots, in case you learn better by reading instead of watching.



##Class 13: Decision Trees

We will look into a slightly more complex model today, the Decision Tree.

**Agenda**

* Slides [here](slides/13_trees.pdf)
* Notebook([here](http://nbviewer.ipython.org/github/sinanuozdemir/SF_DAT_17/blob/master/notebooks/13_decision_trees.ipynb)
* Some more Code [here](code/13_trees.py)

**Homework**

* Project reviews due next Wednesday!

**Resources**

* Chapter 8.1 of An Introduction to Statistical Learning also covers the basics of Classification and Regression Trees
* The scikit-learn [documentation](http://scikit-learn.org/stable/modules/tree.html) has a nice summary of the strengths and weaknesses of Trees.
* For those of you with background in javascript, d3.js has a nice tree layout that would make more presentable tree diagrams:
   * Here is a link to a [static version](http://bl.ocks.org/mbostock/4339184), as well as a link to a [dynamic version](http://bl.ocks.org/mbostock/4339083) with collapsable nodes.
   * If this is something you are interested in, Gary Sieling wrote a nice [function](http://www.garysieling.com/blog/rending-scikit-decision-trees-d3-js) in python to take the output of a scikit-learn tree and convert into json format.
   * If you are intersted in learning d3.js, this a good [tutorial](http://www.d3noob.org/2014/01/tree-diagrams-in-d3js_11.html) for understanding the building blocks of a decision tree. Here is another [tutorial](http://blog.pixelingene.com/2011/07/building-a-tree-diagram-in-d3-js/) focusing on building a tree diagram in d3.js.
* Dr. Justin Esarey from Rice University has a nice [video lecture](https://www.youtube.com/watch?v=HW7Aib842Oo&hd=1) on CART that also includes an [R code walkthrough](http://jee3.web.rice.edu/cart-and-random-forests.r)


### Class 14: Ensembling
* Ensembling ([IPython notebook](http://nbviewer.ipython.org/github/sinanuozdemir/SF_DAT_17/blob/master/notebooks/14_ensembling.ipynb))
* BONUS: Regularization ([IPython notebook](http://nbviewer.ipython.org/github/sinanuozdemir/SF_DAT_17/blob/master/notebooks/14_regularization.ipynb))
	* Bonus advanced sklearn code [here](code/14_advanced_sklearn.py)

**Resources:**

* scikit-learn documentation: [Ensemble Methods](http://scikit-learn.org/stable/modules/ensemble.html)
* Quora: [How do random forests work in layman's terms?](http://www.quora.com/How-do-random-forests-work-in-laymans-terms/answer/Edwin-Chen-1)


### Class 15: Dimension Reduction
* Fun with indeed [here](fun/data_science_indeed.py)
* PCA
    * [Slides](slides/15_dimension_reduction.pdf)
    * Code: [PCA and SVD](code/15_pca_iris.py)
    * Code: [image compression with PCA](code/15_shakira.py) ([original source](http://glowingpython.blogspot.com/2011/07/pca-and-image-compression-with-numpy.html))


*Resources*

* Some hardcore math in python [here](code/15_pca_svd_class.py)
* PCA using the iris data set [here](http://scikit-learn.org/0.11/auto_examples/decomposition/plot_pca_iris.html) and with 2 components [here](http://scikit-learn.org/stable/auto_examples/decomposition/plot_pca_vs_lda.html)
* PCA step by step [here](http://sebastianraschka.com/Articles/2014_pca_step_by_step.html)
* Check out [Pyxley](http://multithreaded.stitchfix.com/blog/2015/07/16/pyxley/)

### Class 16: Neural Networks and SVM

**Agenda**

* slides [here](slides/16_nn_svm.pdf)
* code [here](code/16_nn_svm.py)

* Lab [here](labs/16_lab.md)

**Resources**

* An intro to Neural Networks [here](http://natureofcode.com/book/chapter-10-neural-networks/)
* An intro to [SVM](http://www.cs.columbia.edu/~kathy/cs4701/documents/jason_svm_tutorial.pdf)
* SVM Margins Example [here](http://scikit-learn.org/stable/auto_examples/svm/plot_svm_margin.html)
* SVM digits was adapted from [here](http://pythonprogramming.net/support-vector-machine-svm-example-tutorial-scikit-learn-python/)
* [Google Deep Dream: why does it always see dogs?!](http://www.fastcodesign.com/3048941/why-googles-deep-dream-ai-hallucinates-in-dog-faces)
* [Deep Dream Generator](http://deepdreamgenerator.com/)
* Most used non-sklearn ANN [PyBrain](http://pybrain.org/)
* Step by Step back propagation [here](http://home.agh.edu.pl/~vlsi/AI/backp_t_en/backprop.html)


### Class 18: Recommendation Engines

* Recommendation Engines [slides](slides/19_recommendation_engines.pdf)
* Recommendation Engine Example [code](code/19_recommenders_class.py)

**Resources:**

* [The Netflix Prize](http://www.netflixprize.com/)
* [Why Netflix never implemented the winning solution](https://www.techdirt.com/blog/innovation/articles/20120409/03412518422/why-netflix-never-implemented-algorithm-that-won-netflix-1-million-challenge.shtml)
* [Visualization of the Music Genome Project](http://www.music-map.com/)
* [The People Inside Your Machine](http://www.npr.org/blogs/money/2015/01/30/382657657/episode-600-the-people-inside-your-machine) (23 minutes) is a Planet Money podcast episode about how Amazon Mechanical Turks can assist with recommendation engines (and machine learning in general).


* Next class we will have a talk from three engineers from [OpenGov](http://opengov.com) discussing NLP tactics used for governments around the world!


### Class 19: More Neural Networks

* We will need a new package!   `sudo pip install pybrain`
* Recap [here](slides/16_nn_svm.pdf)
* Let's build our own! [here](code/20_our_nn.py)
* Let's use [Pybrain](http://pybrain.org/)! [here](code/20_pybrain_nn.py)
* A talk from OpenGov

**Resources**

* Code adapted from [here](http://iamtrask.github.io/2015/07/12/basic-python-network/) and [here](http://corpocrat.com/2014/10/10/tutorial-pybrain-neural-network-for-classifying-olivetti-faces/)
* Calculus adapted from [here](http://math.stackexchange.com/questions/78575/derivative-of-sigmoid-function-sigma-x-frac11e-x)
* Sklearn will come out with their own supervised neural network soon! [here](http://scikit-learn.org/dev/modules/neural_networks_supervised.html)


### Class 20: Databases and SQL

* Slides [here](slides/20_db_sql.pdf)
* Code [here](code/20_sql.py)
* Lab [here](labs/20_sql_lab.py) with solutions [here](labs/20_sql_lab_solutions.py)


**Resources**

* This [GA notebook](https://github.com/podopie/DAT18NYC/blob/master/classes/17-relational_databases.ipynb) provides a shorter introduction to databases and SQL that helpfully contrasts SQL queries with Pandas syntax.
* [SQLZOO](http://sqlzoo.net/wiki/SQL_Tutorial), [Mode Analytics](http://sqlschool.modeanalytics.com/), [Khan Academy](https://www.khanacademy.org/computing/computer-programming/sql), [Codecademy](https://www.codecademy.com/courses/learn-sql), [Datamonkey](http://datamonkey.pro/guess_sql/lessons/), and [Code School](http://campus.codeschool.com/courses/try-sql/contents) all have online beginner SQL tutorials that look promising. Code School also offers an [advanced tutorial](https://www.codeschool.com/courses/the-sequel-to-sql/), though it's not free.
* [w3schools](http://www.w3schools.com/sql/trysql.asp?filename=trysql_select_all) has a sample database that allows you to practice SQL from your browser. Similarly, Kaggle allows you to query a large SQLite database of [Reddit Comments](https://www.kaggle.com/c/reddit-comments-may-2015/data) using their online "Scripts" application.
* [What Every Data Scientist Needs to Know about SQL](http://joshualande.com/data-science-sql/) is a brief series of posts about SQL basics, and [Introduction to SQL for Data Scientists](http://bensresearch.com/downloads/SQL.pdf) is a paper with similar goals.
* [10 Easy Steps to a Complete Understanding of SQL](https://web.archive.org/web/20150402234726/http://tech.pro/tutorial/1555/10-easy-steps-to-a-complete-understanding-of-sql) is a good article for those who have some SQL experience and want to understand it at a deeper level.
* SQLite's article on [Query Planning](http://www.sqlite.org/queryplanner.html) explains how SQL queries "work".
* [A Comparison Of Relational Database Management Systems](https://www.digitalocean.com/community/tutorials/sqlite-vs-mysql-vs-postgresql-a-comparison-of-relational-database-management-systems) gives the pros and cons of SQLite, MySQL, and PostgreSQL.
* If you want to go deeper into databases and SQL, Stanford has a well-respected series of [14 mini-courses](https://lagunita.stanford.edu/courses/DB/2014/SelfPaced/about).
* [Blaze](http://blaze.pydata.org) is a Python package enabling you to use Pandas-like syntax to query data living in a variety of data storage systems.

### Next Steps

The hardest thing to do now is to stay sharp! I have a few recommendations on next steps in order to make sure that you don't forget what we learned here! ![](https://s-media-cache-ak0.pinimg.com/originals/88/74/9d/88749d9e7fb7ef330b5e0b4ceb781cfd.jpg)

* Always stay up to date on [Kaggle](http://kaggle.com)
	* Try working with some other people in this class!
	* Our Slack channel will stay around if you still want to post cool blogs, videos, etc!
* Try implementing some of the models we learned in class on your own!
	* Great book [Data Science From Scratch] (http://file.allitebooks.com/20150707/Data%20Science%20from%20Scratch-%20First%20Principles%20with%20Python.pdf) with [code](https://github.com/joelgrus/data-science-from-scratch)
	*  Text classification with Naive Bayes from Scratch [here] (https://web.stanford.edu/class/cs124/lec/naivebayes.pdf)
	*  Introduction to Statistical Learning [book](http://www-bcf.usc.edu/~gareth/ISL/) Videos [here](http://www.dataschool.io/15-hours-of-expert-machine-learning-videos/)
	*  PCA by hand [here](http://sebastianraschka.com/Articles/2014_pca_step_by_step.html)
* Take a look at the **Resources** for each class to get a deeper understanding of what we've learned. Trust me there are a lot
* Follow data scientists on Twitter.  This will help you stay up on the latest news/models/applications/tools.
* Read blogs to keep learning.  I really like [District Data Labs](http://districtdatalabs.silvrback.com/) and [Data Elixer](http://dataelixir.com/).
* There are some active Python Data meetups in the area:
	* [SF Python](http://www.meetup.com/sfpython/)
	* [SF Data Science](http://www.meetup.com/SF-Data-Science/)
	* [SF Data Mining](http://www.meetup.com/Data-Mining/)
	* Request sponsorship for study groups through [GA](https://gaalumni.typeform.com/to/dGyuWb)
	* [General GA Alumni Perks](generalassemb.ly/alumni)

	
Thank you all for such a wonderful time and I truly hope to stay in touch.
