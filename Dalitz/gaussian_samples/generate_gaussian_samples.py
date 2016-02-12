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

if(total==7):
	no_points = int(sys.argv[1])
	x_mean = float(sys.argv[2])
	x_std = float(sys.argv[3])
	y_mean = float(sys.argv[4])
	y_std = float(sys.argv[5])
	label_no =float(sys.argv[6])
else:	
	print("Using standard arguments")
	

x_values = []
while len(x_values) < no_points:
    x_value = gauss(x_mean, x_std)
    if 0 < x_value < 1:
        x_values.append(x_value)

y_values = []
while len(y_values) < no_points:
    y_value = gauss(y_mean, y_std)
    if 0 < y_value < 1:
        y_values.append(y_value)

fo = open("gauss_data/data.+.2Dgauss_"+str(no_points)+"_"+str(x_mean)+"_"+str(x_std)+"_"+str(y_mean)+"_"+str(y_std)+"_"+str(label_no)+  ".txt", "w")

for i in range(len(x_values)):
	print(str(x_values[i]) + "  " + str(y_values[i]), file=fo)

#print(values)
