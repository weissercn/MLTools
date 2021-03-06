#!/usr/bin/env python

#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
#
# Name:     goodness_of_fit.py 
#
# Author:   Constantin Weisser (weisser@mit.edu)
#
# Purpose:  This is the main file of the Dalitz hirarchy. It takes in two 
#	    data files and uses either classifier_eval (scikit learn) 
#	    or tensorflow to compare these two files. 
#
#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#

from __future__ import print_function
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np
import math
from sklearn import preprocessing
from sklearn import tree
from sklearn.ensemble import AdaBoostClassifier
from sklearn import svm
from sklearn import cross_validation
from sknn.mlp import Classifier, Layer
from scipy import stats
import sys
#sys.path.append("$MLToolsDir")
sys.path.append("../..")
import classifier_eval

np.random.seed(100) 

"""
================================================================
PREPARING THE DATA
================================================================
"""
print(__doc__)

#this will be overwritten
name="2Dgauss_1000"
sample1_name="mean_0_5"
sample2_name="mean_0_48"

#file 0 contains the particle, file 1 the antiparticle samples.
#comp_file_0='gaussian_samples/gauss_data/data.2Dgauss_1000_0.5_0.1_0.5_0.1_1.0.txt'
#comp_file_1='gaussian_samples/gauss_data/data.2Dgauss_1000_0.48_0.1_0.48_0.1_1.0.txt'

comp_file_0='data.+.txt'
comp_file_1='data.cpv.txt'

import sys

#Extracting arguments 
args = str(sys.argv)
total = len(sys.argv)

if(total!=7):
	print("The right arguments were not supplied")

comp_file_0    = str(sys.argv[1])
comp_file_1    = str(sys.argv[2])
name           = str(sys.argv[3])
sample1_name   = str(sys.argv[4])
sample2_name   = str(sys.argv[5])
shuffling_seed = int(sys.argv[6])
np.random.seed(shuffling_seed)

print(comp_file_0)
print(comp_file_1)
print(name)
print(sample1_name)
print(sample2_name)
print(shuffling_seed)


#extracts data from the files
features_0=np.loadtxt(comp_file_0,dtype='d')
features_1=np.loadtxt(comp_file_1,dtype='d')

#determine how many data points are in each sample
no_0=features_0.shape[0]
no_1=features_1.shape[0]

#Give all samples in file 0 the label 0 and in file 1 the feature 1
label_0=np.zeros((no_0,1))
label_1=np.ones((no_1,1))

#Create an array containing samples and features.
data_0=np.c_[features_0,label_0]
data_1=np.c_[features_1,label_1]

data=np.r_[data_0,data_1]

#USING STANDARD SCALER TO REMOVE MEAN AND STANDARD DEVIATION
#data[:,:-1]=preprocessing.StandardScaler().fit_transform(data[:,:-1])
#This should be done within the class

np.savetxt('data_unshuffled.txt', data)
#Shuffle data
np.random.shuffle(data)

np.savetxt('data.txt', data)

#dt_example=classifier_eval.dt_sklearn(data,40,0,name,sample1_name,sample2_name)
#dt_example.get_results()

#ada_example=classifier_eval.ada_sklearn(data,40,0,1000,name,sample1_name,sample2_name)
#ada_example.get_results()

#svm_example=classifier_eval.svm_sklearn(data,40,0,1000,name,sample1_name,sample2_name)
#svm_example.get_results()

#nn_example=classifier_eval.nn_sklearn(data,40,0,name,sample1_name,sample2_name)
#nn_example.get_results()

#softmax_example=classifier_eval.softmax_regression_tf(data,40,2)
#softmax_example.get_results()

#keras_example=classifier_eval.keras_classifier(data,40,0,name,sample1_name,sample2_name)
#keras_example.get_results()

#miranda_example=classifier_eval.twodim_miranda(data,40,0,10,10,name,sample1_name,sample2_name)
#miranda_example.get_results()

#energy_test_example=classifier_eval.twodim_energy_test(data,40,0,0.15,features_0,features_1,name,sample1_name,sample2_name)
#energy_test_example.get_results()

energy_test_example=classifier_eval.twodim_energy_test_C(data,40,0,0.15,features_0,features_1,name,sample1_name,sample2_name)
energy_test_example.get_results()
