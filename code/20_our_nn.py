import numpy as np

# sigmoid function
#EXERCISE
# What should be in the deriv part?
def nonlin(x,deriv=False):
    if(deriv==True):
        return 0.0 #BLANK
    return 1/(1+np.exp(-x))
# Note the deriv is not what we calculated. When we use the deriv function
# We are actually inputting f(x) making the derivative correct    
    
# input dataset
X = np.array([  [0,0,1],
                [0,1,1],
                [1,0,1],
                [1,1,1] ])
    
# output dataset            
y = np.array([[0,0,1,1]]).T

# seed random numbers to make calculation
# deterministic (just a good practice)
np.random.seed(1)
'''
Initialize weights randomly with mean 0
Note that it is a 3 x 1 matrix. This is because
we have two layers, an input of 3 dimensions and an output of 
one dimenension. IF we had more than two layers, we would 
need more than one matrix
'''
syn0 = 2*np.random.random((3,1)) - 1
# the 2* () - 1 is to make it between -1 and 1

syn0

'''
Again these represent our initial weights, 
if syn0 == 
array([[-0.16595599],
       [ 0.44064899],
       [-0.99977125]])
Then initially our first observation would be:
-.17*0 + .44 * 0 + -.99 * 1 = - 0.99
then we would do nonlin(-0.99) = 0.2709 giving us a 
signal (response) of 0.27
Our l1_error would be -0.27 = (actual - prediction) = (0 - 0.27)
our l1_delta is the error time the slope on the sigmoid function
    l1_error * nonlin(- 0.99, True) = -0.064
so our first example gives the signal to roll back on the weights 
by 0.064 downwards.


'''
# the 10000 represents 10k "epochs"
# This is a full batch configuration.
# Each training set gets put into the network as a full batch
# 10,000 times
for iter in xrange(10000):

    # forward propagation
    l0 = X
    weighted_sum = np.dot(l0,syn0)
    l1 = nonlin(weighted_sum)
    # l1 represents our guess at each "epoch"
    # our predictions

    # how much did we get it wrong by?
    # EXERCISE
    # l1_error represents the difference between 
    # the actual - predicted error
    l1_error = 0.0#BLANK

    if iter % 1000 == 0:
        preds = np.where(l1 > .5, 1, 0)
        print "Accuracy", float(sum(preds == y)) / len(y)
    # multiply how much we missed by the 
    # slope of the sigmoid at the values in l1
    l1_delta = l1_error * nonlin(weighted_sum,True)
    # This is the "The Error Weighted Derivative"

    # update weights using our error weighted derivative
    syn0 += np.dot(l0.T,l1_delta)

print "Output After Training:"
print l1

print syn0

