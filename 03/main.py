import argparse
from time import time

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split

import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader

from model import MultiLayerPerceptron
from dataset import AdultDataset
from util import *


""" Adult income classification

In this lab we will build our own neural network pipeline to do classification on the adult income dataset. More
information on the dataset can be found here: http://www.cs.toronto.edu/~delve/data/adult/adultDetail.html

"""
seed = 0

# =================================== LOAD DATASET =========================================== #

######

# 3.1 YOUR CODE HERE

######

# =================================== DATA VISUALIZATION =========================================== #

# the dataset is imported as a DataFrame object, for more information refer to
# https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.html
# we can check the number of rows and columns in the dataset using the .shape field
# to get a taste of what our datset looks like, let's visualize the first 5 rows of the dataset using the .head() method
# the task will be to predict the "income" field (>50k or <50k) based on the other fields in the dataset
# check how balanced our dataset is using the .value_counts() method.

######

# 3.2 YOUR CODE HERE

######


# =================================== DATA CLEANING =========================================== #

# datasets often come with missing or null values, this is an inherent limit of the data collecting process
# before we run any algorithm, we should clean the data of any missing values or unwanted outliers which could be
# detrimental to the performance or training of the algorithm. In this case, we are told that missing values are
# indicated with the symbol "?" in the dataset

# let's first count how many missing entries there are for each feature
col_names = data.columns
num_rows = data.shape[0]
for feature in col_names:
    pass
    ######

    # 3.3 YOUR CODE HERE

    ######

# next let's throw out all rows (samples) with 1 or more "?"
# Hint: take a look at what data[data["income"] != ">50K"] returns
# Hint: if data[field] do not contain strings then data["income"] != ">50K" will return an error

    ######

    # 3.3 YOUR CODE HERE

    ######

# =================================== BALANCE DATASET =========================================== #

    ######

    # 3.4 YOUR CODE HERE

    ######

# =================================== DATA STATISTICS =========================================== #

# our dataset contains both continuous and categorical features. In order to understand our continuous features better,
# we can compute the distribution statistics (e.g. mean, variance) of the features using the .describe() method

######

# 3.5 YOUR CODE HERE

######

# likewise, let's try to understand the distribution of values for discrete features. More specifically, we can check
# each possible value of a categorical feature and how often it occurs
categorical_feats = ['workclass', 'race', 'education', 'marital-status', 'occupation',
                    'relationship', 'gender', 'native-country', 'income']

for feature in categorical_feats:
    pass
    ######

    # 3.5 YOUR CODE HERE

    ######

# visualize the first 3 features using pie and bar graphs

######

# 3.5 YOUR CODE HERE

######

# =================================== DATA PREPROCESSING =========================================== #

# we need to represent our categorical features as 1-hot encodings
# we begin by converting the string values into integers using the LabelEncoder class
# next we convert the integer representations into 1-hot encodings using the OneHotEncoder class
# we don't want to convert 'income' into 1-hot so let's extract this field first
# we also need to preprocess the continuous features by normalizing against the feature mean and standard deviation
# don't forget to stitch continuous and cat features together

# ENCODE CATEGORICAL FEATURES

# Helpful Hint: .values converts the DataFrame to a numpy array

# LabelEncoder information: https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelEncoder.html
#       the LabelEncoder works by transforming the values in an input feature into a 0-to-"n_classes-1" digit label 
#       if a feature in the data has string values "A, B, X, Y", then LabelEncoder will turn these into the numeric 0, 1, 2, 3
#       like other scikit-learn objects, the LabelEncoder must first fit() on the target feature data (does not return anything)
#       fitting on the target feature data creates the mapping between the string values and the numerical labels
#       after fitting, then transform() on a set of target feature data will return the numerical labels representing that data
#       the combined fit_transform() does this all in one step. Check the examples in the doc link above!

label_encoder = LabelEncoder()
######

# 3.6 YOUR CODE HERE
labelencoder = LabelEncoder()
for feature in categorical_feats:
    pass # replace with code that converts each string-valued feature into a numeric feature using the LabelEncoder

######

# OneHotEncoder information: https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html
#    the OneHotEncoder works basically identical to the LabelEncoder
#    however, its input, instead of a single numeric array, is a matrix (dense or sparse) of 0 and 1 values
#    consider the following tabular data X of N data points (assume it is a data frame):
#
#    F1     F2      F3
#    ON     1     Toronto
#    ON     3     Scarborough
#    OFF    2     North York
#    ON     3     Toronto
#    OFF    3     Etobicoke
#    OFF    1     Scarborough
#    ...
#
#    F1 has 2 string values (ON, OFF), F2 has 2 numeric values (1, 2, 3), and F3 has 4 string values (Toronto, Scarborough, 
#       North York, Etobicoke)
#    When we use the OneHotEncoder's fit_transform on this data frame X, the resulting matrix takes the shape: N x (2 + 3 + 4)
#
#    [[1 0 1 0 0 1 0 0 0]
#     [1 0 0 0 1 0 1 0 0]
#     [0 1 0 1 0 0 0 1 0]
#     [1 0 0 0 1 1 0 0 0]
#     [0 1 0 0 1 0 0 0 1]
#     [0 1 1 0 0 0 1 0 0]
#    ...
#
#    In other words, for tabular data with N data points and k features F1 ... Fk,
#    Then the resulting output matrix will be of size (N x (F1_n + ... + Fk_n))
#    This is because, looking at datapoint 2 for example: [1 0 0 0 1 0 1 0 0],
#    [1 0 | 0 0 1 | 0 1 0 0] -> here, [1 0] is the encoding for "ON" (ON vs OFF), [0 0 1] is the encoding for "3" (1 vs 2 vs 3), etc.
#    If a single _categorical variable_ F has values 0 ... N-1, then its 1-of-K encoding will be a vector of length F_n
#    where all entries are 0 except the value the data point takes for F at that point, which is 1.
#    Thus, for features F1 ... Fk, as described above, the length-Fi_n encodings are appended horizontally.

# firstly, we need to drop 'income' becaue we don't want to convert it into one-hot encoding:
y = data['income']
data = data.drop(columns=['income'])
categorical_feats.remove('income')
y = y.values  # convert DataFrame to numpy array


# now, we can use the OneHotEncoder on the part of the data frame encompassed by 'categorial_feats'
# we can fit and transform as usual. Your final output one-hot matrix should be in the variable 'cat_onehot'
oneh_encoder = OneHotEncoder()
######

# 3.6 YOUR CODE HERE

######
cat_onehot = np.zeros((2,2)) # replace this with the output of your fit and transform

# NORMALIZE CONTINUOUS FEATURES

# finally, we need to preprocess the continuous features by normalizing against the feature mean and standard deviation
# we begin by storing the data dropped of the categorical_feats in a separate variable, 'cts_data'
# your task is to use .mean() and .std() on this data to normalize it, then covert it into a numpy array

cts_data = data.drop(columns=categorical_feats)
######

# 3.6 YOUR CODE HERE

######

# finally, we stitch continuous and categorical features
X = np.concatenate([cts_data, cat_onehot], axis=1)
print("Shape of X =", X.shape)

# =================================== MAKE THE TRAIN AND VAL SPLIT =========================================== #
# we'll make use of the train_test_split method to randomly divide our dataset into two portions
# control the relative sizes of the two splits using the test_size parameter

######

# 3.7 YOUR CODE HERE

######

# =================================== LOAD DATA AND MODEL =========================================== #

def load_data(batch_size):
    ######

    # 4.1 YOUR CODE HERE

    ######


    return train_loader, val_loader


def load_model(lr):

    ######

    # 4.4 YOUR CODE HERE

    ######

    return model, loss_fnc, optimizer


def evaluate(model, val_loader):
    total_corr = 0

    ######

    # 4.6 YOUR CODE HERE

    ######

    return float(total_corr)/len(val_loader.dataset)


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--lr', type=float)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--eval_every', type=int, default=10)

    args = parser.parse_args()

    ######

    # 4.5 YOUR CODE HERE

    ######


if __name__ == "__main__":
    main()
