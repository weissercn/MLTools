#!/usr/local/bin/python

#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
#
# Name:     /Dalitz/automation_scripts/optimisation/svm/automate_goodness_of_fit.py 
#
# Author:   Constantin Weisser (weisser@mit.edu)
#
# Purpose:  This is a python script to automate analysing data files
#
#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-# 

from __future__ import print_function
from __future__ import division
import os
import math

number_of_files=1

name           = "dalitz_gamma_optimisation"
sample1_name   = "seed_000_0"
# You might have to change this to "seed_100_0"
sample2_name   = "seed_200_1"
shuffling_seed = 100
# You can chose from dectree, adaboost svm neuralnet softmax keras miranda etest or any combination of these by connecting them with underscores
classifier_name= "svm"

#min_samples_split_options=["auto","sqrt","log2",None]
#min_samples_split_options=["best","random"]
hyperparameter_options= [x / 20 for x in range(1,21)]

svm_C=0.4
coef0=0.0
gamma="auto"

#Doing the loop. #-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#  


for i in range(1,number_of_files+1):
	for gamma in hyperparameter_options:
		os.system("python -O $MLToolsDir/Dalitz/goodness_of_fit.py $MLToolsDir/Dalitz/dpmodel/data/data.{0}.0.txt $MLToolsDir/Dalitz/dpmodel/data/data.2{1}.1.txt {2} {3} {4} {5} {6} {7} {8} {9}".format(i-1,str(i-1).zfill(2),name,sample1_name,sample2_name,shuffling_seed,classifier_name,svm_C,coef0,gamma))




