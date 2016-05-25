#!/usr/bin/env python

#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
#
# Name:     generate_gaussian_samples.py 
#
# Author:   Constantin Weisser (weisser@mit.edu)
#
# Purpose:  This is a python script to write a file containing 10000 data points
#	    sampled from a 2D Gaussian
#
#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-# 


#Constantin Weisser
from __future__ import print_function
from random import gauss
import sys
import numpy as np
import json		#needed to read in means and stdev as numpy arrays
import random

no_points=10000
original_mean1 = 0.2
original_mean2 = 0.8
original_std = 0.05 
label_no = 1

args = str(sys.argv)
#print ("Args list: %s " % args)
#The first argument is the name of this python file
total = len(sys.argv)
verbose=True

if(total==8):
	no_points = int(sys.argv[1])
	#mean = np.array(json.loads(sys.argv[2]))
	#std = np.array(json.loads(sys.argv[3]))
	original_mean1 = float(sys.argv[2])
	original_mean2 = float(sys.argv[3])
	original_std = float(sys.argv[4])
	distance_to_original = float(sys.argv[5])
	no_dim = int(sys.argv[6])
	label_no =float(sys.argv[7])
else:	
	print("Using standard arguments")

if verbose:
	print("original_mean1 : ", original_mean1)
	print("original_mean2 : ", original_mean2)
	print("original_std : ",original_std)


#print(mean.shape[0])

for dim in range(no_dim):
	values = np.zeros((no_points,1))
	for i in range(no_points):
		if bool(random.getrandbits(1)):
			values[i] = gauss(original_mean1+distance_to_original,original_std)
		else:
			values[i] = gauss(original_mean2-distance_to_original,original_std)
	#print(values)
	if dim==0:
		full_cords=values
	else:
		full_cords=np.column_stack((full_cords,values))


print(full_cords)

np.savetxt("gauss_data/data_double_high{0}Dgauss_".format(int(no_dim))+str(int(no_points))+"_"+str(original_mean1)+"_"+str(original_mean2)+"_"+str(original_std)+"_"+str(distance_to_original)+"_"+str(int(label_no))+  ".txt",full_cords)





