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

name           = "gaussian_samples"
sample1_name   = "mean_0_5"
# You might have to change this to "seed_100_0"
sample2_name   = "mean_0_48"
shuffling_seed = 100

#with open("test_statistic_distributions/test_statistics."+name+"_"+sample1_name+"_"+sample2_name, "w") as test_statistics_file:
	#test_statistics_file.write("CvM U \tCvM T \tKS D \tKS p\n")

for dim in range(1,11):
	for i in range(1,number_of_files+1):
		# python -O sets __debug__ to 0
		#os.system("touch test_statistic_distributions/test_statistics.{0}_{1}_{2}".format(name,sample1_name,sample2_name))
		os.system("python -O goodness_of_fit.py $MLToolsDir/Dalitz/gaussian_samples/higher_dimensional_gauss/gauss_data/data_high{5}Dgauss_10000_0.5_0.1_0.0_{0}.txt $MLToolsDir/Dalitz/gaussian_samples/higher_dimensional_gauss/gauss_data/data_high{5}Dgauss_10000_0.5_0.1_0.01_{0}.txt {1} {2} {3} {4}".format(i,name,sample1_name,sample2_name,shuffling_seed,dim))
		#You might want to change data.2{1}.1.txt to data.1{1}.0.txt
		#os.system("python -O goodness_of_fit.py ../../dpmodel/data/data.{0}.0.txt ../../dpmodel/data/data.2{1}.1.txt {2} {3} {4} {5}".format(i-1,str(i-1).zfill(2),name,sample1_name,sample2_name,shuffling_seed))




