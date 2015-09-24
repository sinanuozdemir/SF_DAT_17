'''
Move this code into your OWN SF_DAT_15_WORK repo

Please complete each question using 100% python code

If you have any questions, ask a peer or one of the instructors!

When you are done, add, commit, and push up to your repo

This is due 7/1/2015
'''


import pandas as pd
# pd.set_option('max_colwidth', 50)
# set this if you need to

killings = pd.read_csv('hw/data/police-killings.csv')
killings.head()

# 1. Make the following changed to column names:
# lawenforcementagency -> agency
# raceethnicity        -> race

# 2. Show the count of missing values in each column

# 3. replace each null value in the dataframe with the string "Unknown"

# 4. How many killings were there so far in 2015?

# 5. Of all killings, how many were male and how many female?

# 6. How many killings were of unarmed people?

# 7. What percentage of all killings were unarmed?

# 8. What are the 5 states with the most killings?

# 9. Show a value counts of deaths for each race

# 10. Display a histogram of ages of all killings

# 11. Show 6 histograms of ages by race

# 12. What is the average age of death by race?

# 13. Show a bar chart with counts of deaths every month



###################
### Less Morbid ###
###################

majors = pd.read_csv('hw/data/college-majors.csv')
majors.head()

# 1. Delete the columns (employed_full_time_year_round, major_code)

# 2. Show the cout of missing values in each column

# 3. What are the top 10 highest paying majors?

# 4. Plot the data from the last question in a bar chart, include proper title, and labels!


# 5. What is the average median salary for each major category?

# 6. Show only the top 5 paying major categories

# 7. Plot a histogram of the distribution of median salaries

# 8. Plot a histogram of the distribution of median salaries by major category

# 9. What are the top 10 most UNemployed majors?
# What are the unemployment rates?

# 10. What are the top 10 most UNemployed majors CATEGORIES? Use the mean for each category
# What are the unemployment rates?

# 11. the total and employed column refer to the people that were surveyed.
# Create a new column showing the emlpoyment rate of the people surveyed for each major
# call it "sample_employment_rate"
# Example the first row has total: 128148 and employed: 90245. it's 
# sample_employment_rate should be 90245.0 / 128148.0 = .7042

# 12. Create a "sample_unemployment_rate" colun
# this column should be 1 - "sample_employment_rate"
