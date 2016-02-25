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

number_of_files=2

name           = "2Dgauss_1000"
sample1_name   = "mean_0_5"
sample2_name   = "mean_0_48"
shuffling_seed = 100

with open("test_statistic_distributions/test_statistics."+name+"_"+sample1_name+"_"+sample2_name, "w") as test_statistics_file:
	test_statistics_file.write("CvM U \tCvM T \tKS D \tKS p\n")

for i in range(1,number_of_files+1):
	#os.system("touch test_statistic_distributions/test_statistics.{0}_{1}_{2}".format(name,sample1_name,sample2_name))
	os.system("python goodness_of_fit.py gaussian_samples/gauss_data/data.2Dgauss_1000_0.5_0.1_0.5_0.1_{0}.0.txt gaussian_samples/gauss_data/data.2Dgauss_1000_0.48_0.1_0.48_0.1_{0}.0.txt {1} {2} {3} {4}".format(i,name,sample1_name,sample2_name,shuffling_seed))





