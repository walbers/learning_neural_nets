# https://towardsdatascience.com/neural-networks-from-scratch-easy-vs-hard-b26ddc2e89c7
# pics of 0-9
# data set from sklearn
# pics ares 8x8 w each pixel having 0-15 different possibilies of color
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
matplotlib.use('Agg')


dig = load_digits()
onehot_target = pd.get_dummies(dig.target) # if number is four then [0,0,0,0,1,0,0,0,0,0]
x_train, x_val, y_train, y_val = train_test_split(dig.data, onehot_target, test_size=0.1, random_state=20) # splits data set into training and testing
# x_train (1617, 64)
# y_train (1617, 10)

# input layer - 64 neurons becuz image is 8x8

# hidden layer - 1 and 2 have 128 neurons, arbitrary - sigmoid activation

# output layer - 10 nodes because number can be 0-9 - softmax error?
