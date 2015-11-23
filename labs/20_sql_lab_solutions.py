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
create a list called tables that holds the 
two tables in the database
'''
cur = con.cursor()    
cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
tables = cur.fetchall()

'''
create a dataframe called train that has all columns from the 
training table
'''
train = pd.read_sql_query('SELECT * FROM vehicle_train', con=con)

# SANITY CHECK
train.shape # should == (14, 5)
# END SANITY CHECK


# Encode car as 0 and truck as 1 in the train database
train['type'] = train.type.map({'car':0, 'truck':1})

'''
Create a list of the feature columns called feature_cols 
(every column except for the 0th column)
'''
feature_cols = train.columns[1:]

# Define X (features) and y (response)
X = train[feature_cols]
y = train.price


'''
Create a fit a decision tree (with random_state=1)
using the training and test data from above, for now do
not worry about train_test_split
 '''
from sklearn.tree import DecisionTreeRegressor
treereg = DecisionTreeRegressor(random_state=1)
treereg.fit(X, y)


'''
Use 3-fold cross-validation to estimate the RMSE for this model
Divide the final RMSE by the average price to get a more
general view
'''

from sklearn.cross_validation import cross_val_score
import numpy as np
scores = cross_val_score(treereg, X, y, cv=3, scoring='mean_squared_error')
RMSE = np.mean(np.sqrt(-scores))
RMSE / np.mean(y)



##### Testing #####

# Get testing data from database, call the new dataframe: test
test = pd.read_sql_query('SELECT * FROM vehicle_test', con=con)

# Encode car as 0 and truck as 1
test['type'] = test.type.map({'car':0, 'truck':1})

# Define X_test and y_test by the above feature_cols
X_test = test[feature_cols]
y_test = test.price

# Make predictions on test data
y_pred = treereg.predict(X_test)

# Calculate RMSE
from sklearn import metrics
np.sqrt(metrics.mean_squared_error(y_test, y_pred))

#########################
#### More Sales Data ####
#########################

con = lite.connect('../data/sales.db')

orders = pd.read_sql_query("""SELECT * FROM Orders""", con=con)
customers = pd.read_sql_query("""SELECT * FROM Customers""", con=con)
details = pd.read_sql_query("""SELECT * FROM OrderDetails""", con=con)

# Perform a left join and orders and customers
pd.read_sql_query("""SELECT * 
                    FROM Orders 
                    INNER JOIN Customers 
                    ON Orders.customerID = Customers.customerID"""
                    , con)
                    
# Perform a left join and orders and customers for customers in TX
pd.read_sql_query("""SELECT * 
                    FROM Orders 
                    INNER JOIN Customers 
                    ON Orders.customerID = Customers.customerID
                    WHERE Customers.State == "TX"
                    """
                    , con)
                    
# get the total price for each order using the orderdetails table
pd.read_sql_query("""SELECT OrderID, Sum(UnitPrice) as total_price
                FROM OrderDetails GROUP BY OrderID 
                """
                , con)
                


                    
