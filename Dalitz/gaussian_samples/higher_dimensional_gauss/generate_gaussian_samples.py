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

no_points=10000
x_mean=0.5
x_std=0.1
y_mean=0.5
y_std=0.1
label_no = 1

args = str(sys.argv)
#print ("Args list: %s " % args)
#The first argument is the name of this python file
total = len(sys.argv)

verbose=False

if(total==7):
	no_points = int(sys.argv[1])
	#mean = np.array(json.loads(sys.argv[2]))
	#std = np.array(json.loads(sys.argv[3]))
	original_mean = float(sys.argv[2])
	original_std = float(sys.argv[3])
	distance_of_centers = float(sys.argv[4])
	no_dim = int(sys.argv[5])

	label_no =float(sys.argv[6])
else:	
	print("Using standard arguments")

if verbose:
	print(original_mean)
	print(distance_of_centers)

distance_oneD=distance_of_centers/np.sqrt(no_dim)
if verbose:
	print(distance_oneD)

mean=np.empty(no_dim)
mean.fill(original_mean-distance_oneD)
if verbose:
	print(mean)

std=np.empty(no_dim)
std.fill(original_std)
if verbose:
	print(std)



#print(mean.shape[0])

for dim in range(mean.shape[0]):
	values = np.zeros((no_points,1))
	for i in range(no_points):
		values[i] = gauss(mean[dim],std[dim])
	#print(values)
	if dim==0:
		full_cords=values
	else:
		full_cords=np.column_stack((full_cords,values))


print(full_cords)

np.savetxt("gauss_data/data_high{0}Dgauss_optimisation_".format(int(no_dim))+str(int(no_points))+"_"+str(original_mean)+"_"+str(original_std)+"_"+str(distance_of_centers)+"_"+str(int(label_no))+  ".txt",full_cords)
#fo = open("gauss_data/data.manyDgauss_"+str(no_points)+"_"+str(x_mean)+"_"+str(x_std)+"_"+str(y_mean)+"_"+str(y_std)+"_"+str(label_no)+  ".txt", "w")

#for i in range(len(x_values)):
	#print(str(x_values[i]) + "  " + str(y_values[i]), file=fo)

#print(values)
