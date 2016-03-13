#!/usr/local/bin/python

#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
#
# Name:     automate_goodness_of_fit.py 
#
# Author:   Constantin Weisser (weisser@mit.edu)
#
# Purpose:  This is a python script to automate analysing data files
#
#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-# 

from __future__ import print_function
import os
import math

number_of_files=100

name           = "dalitz"
sample1_name   = "seed_000_0"
# You might have to change this to "seed_100_0"
sample2_name   = "seed_200_1"
shuffling_seed = 100
# You can chose from dectree, adaboost svm neuralnet softmax keras miranda etest or any combination of these by connecting them with underscores
classifier_name= "keras"


#with open("test_statistic_distributions/test_statistics."+name+"_"+sample1_name+"_"+sample2_name, "w") as test_statistics_file:
	#test_statistics_file.write("CvM U \tCvM T \tKS D \tKS p\n")

for i in range(1,number_of_files+1):
	# python -O sets __debug__ to 0
	#os.system("touch test_statistic_distributions/test_statistics.{0}_{1}_{2}".format(name,sample1_name,sample2_name))
	#os.system("python -O goodness_of_fit.py gaussian_samples/gauss_data/data.2Dgauss_1000_0.5_0.1_0.5_0.1_{0}.0.txt gaussian_samples/gauss_data/data.2Dgauss_1000_0.48_0.1_0.48_0.1_{0}.0.txt {1} {2} {3} {4}".format(i,name,sample1_name,sample2_name,shuffling_seed))
	#You might want to change data.2{1}.1.txt to data.1{1}.0.txt
	os.system("python -O $MLToolsDir/Dalitz/goodness_of_fit.py ../../dpmodel/data/data.{0}.0.txt ../../dpmodel/data/data.2{1}.1.txt {2} {3} {4} {5} {6}".format(i-1,str(i-1).zfill(2),name,sample1_name,sample2_name,shuffling_seed,classifier_name))




