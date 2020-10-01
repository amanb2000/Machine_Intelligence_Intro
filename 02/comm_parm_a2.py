# Assignment 2 skeleton code
# This code shows you how to use the 'argparse' library to read in parameters

import argparse

# Command Line Arguments
parser = argparse.ArgumentParser(description='generate training and validation data for assignment 2')
parser.add_argument('trainingfile', help='name stub for training data and label output in csv format',default="train")
parser.add_argument('validationfile', help='name stub for validation data and label output in csv format',default="valid")
parser.add_argument('numtrain', help='number of training samples',type= int,default=200)
parser.add_argument('numvalid', help='number of validation samples',type= int,default=20)
parser.add_argument('-seed', help='random seed', type= int,default=1)
parser.add_argument('-learningrate', help='learning rate', type= float,default=0.1)
parser.add_argument('-actfunction', help='activation functions', choices=['sigmoid', 'relu', 'linear'],default='linear')
parser.add_argument('-numepoch', help='number of epochs', type= int,default=50)

args = parser.parse_args()

traindataname = args.trainingfile + "data.csv"
trainlabelname = args.trainingfile + "label.csv"

print("training data file name: ", traindataname)
print("training label file name: ", trainlabelname)

validdataname = args.validationfile + "data.csv"
validlabelname = args.validationfile + "label.csv"

print("validation data file name: ", validdataname)
print("validation label file name: ", validlabelname)

print("number of training samples = ", args.numtrain)
print("number of validation samples = ", args.numvalid)

print("learning rate = ", args.learningrate)
print("number of epoch = ", args.numepoch)

print("activation function is ",args.actfunction)
