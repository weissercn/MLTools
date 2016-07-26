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



no_points=10000
original_mean=0.0
original_std=1.0
is_MC = 1

if is_MC == 0:
	distance_of_centers=0.0 
	distance_of_centers_std = 0.5
	number_of_files=1
	name = "systematics_Data"
else:
        distance_of_centers=3.0 
        distance_of_centers_std = 0.0
	number_of_files=100
	name = "systematics_MC"

log_every_n = number_of_files // 10

for no_dim in range(1,11):

	for i in range(number_of_files):
	#for i in range(1,number_of_files+1):
		#if(i%log_every_n ==0):
			#print("{0} files have been written so far".format(i))
		#os.system("./generate_gaussian_samples.py 10000 [0.48,0.48,0.48] [0.1,0.1,0.1] {0}".format(int(i))) 
		command ="./generate_gaussian_samples.py {0} {1} {2} {3} {4} {5} {6} {7}".format(int(no_points),float(original_mean),float(original_std),float(distance_of_centers),int(no_dim    ),int(i),float(distance_of_centers_std),name)
		print(command)
		os.system(command)
		#os.system("./generate_gaussian_samples.py {0} {1} {2} {3} {4} {5} {6}".format(int(no_points),float(original_mean),float(original_std),float(distance_of_centers),int(no_dim),int(i)))

	print("{0} files have been generated in dimension {1}".format(number_of_files,no_dim))


os.system('say "Done generating samples."');
