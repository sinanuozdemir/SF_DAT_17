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
11/16: Recommendation Engines | 11/18: TBD
11/24: TBD | 11/26: TBD
11/31: Projects | 12/2: Projects


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