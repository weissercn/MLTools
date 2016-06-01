#!/usr/local/bin/python

#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
#
# Name:     automate_generate_gaussian_samples.py 
#
# Author:   Constantin Weisser (weisser@mit.edu)
#
# Purpose:  This is a python script to automate writing file containing data points
#           sampled from a 2D Gaussian
#
#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-# 

from __future__ import print_function
from __future__ import division
import os
import math
import numpy as np


number_of_files=100
log_every_n = number_of_files // 10

no_points=10000
original_mean1=0.25
original_mean2=0.75
original_std=0.1
distance_to_original = 0.02
#use 0.00 for the standard sample and say 0.03 for the cpv sample

for no_dim in range(2,11):

	for i in range(number_of_files):
	#for i in range(1,number_of_files+1):
		#if(i%log_every_n ==0):
			#print("{0} files have been written so far".format(i))
		#os.system("./generate_gaussian_samples.py 10000 [0.48,0.48,0.48] [0.1,0.1,0.1] {0}".format(int(i))) 
			
		command ="./generate_gaussian_samples.py {0} {1} {2} {3} {4} {5} {6}".format(int(no_points),float(original_mean1),float(original_mean2),float(original_std),float(distance_to_original),int(no_dim),int(i))
		print(command)
		os.system(command)
		#os.system("./generate_gaussian_samples.py {0} {1} {2} {3} {4} {5} {6}".format(int(no_points),float(original_mean),float(original_std),float(distance_of_centers),int(no_dim),int(i)))

	print("{0} files have been generated".format(number_of_files))

os.system('say "Done generating samples."');
