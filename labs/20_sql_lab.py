import sqlite3 as lite
import pandas as pd

###############################################################################
##### Normal Data Science Process with a Database
###############################################################################

"""
Finally, just to reiterate that getting data from databases is nothing more
than another way to get data (and thus, has no effect upon the rest of the data
science process), here is some code we used in a previous class.  Instead of 
reading data from a CSV file, we get it from a database.
"""

##### Training #####

# Open new connection
con = lite.connect('../data/vehicles.db')

'''
1. create a list called tables that holds the 
two tables in the database
'''

'''
2. create a dataframe called train that has all columns from the 
training table
'''

# SANITY CHECK
train.shape # should == (14, 5)
# END SANITY CHECK


# 3. Encode car as 0 and truck as 1 in the train database

'''
4. Create a list of the feature columns called feature_cols 
(every column except for the 0th column)
'''

# Define X (features) and y (response)
X = train[feature_cols]
y = train.price


'''
5. Create a fit a decision tree (with random_state=1)
using the training and test data from above, for now do
not worry about train_test_split
 '''
from sklearn.tree import DecisionTreeRegressor


'''
6. Use 3-fold cross-validation to estimate the RMSE for this model
Divide the final RMSE by the average price to get a more
general view
'''

from sklearn.cross_validation import cross_val_score


##### Testing #####

# 7. Get testing data from database, call the new dataframe: test

# 8. Encode car as 0 and truck as 1

# Define X_test and y_test by the above feature_cols
X_test = test[feature_cols]
y_test = test.price

# 9. Make predictions on test data

# 10. Calculate RMSE


#########################
#### More Sales Data ####
#########################

con = lite.connect('../data/sales.db')

orders = pd.read_sql_query("""SELECT * FROM Orders""", con=con)
customers = pd.read_sql_query("""SELECT * FROM Customers""", con=con)
details = pd.read_sql_query("""SELECT * FROM OrderDetails""", con=con)

# Perform a left join and orders and customers

# Perform a left join and orders and customers for customers in TX
                    
# get the total price for each order using the orderdetails table
                


                    
