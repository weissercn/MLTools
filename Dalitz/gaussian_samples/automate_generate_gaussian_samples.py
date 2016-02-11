#!/usr/local/bin/python

from __future__ import print_function
import os



number_of_files=1000
log_every_n = number_of_files // 10

for i in range(1,number_of_files):
	if(i%log_every_n ==0):
		print("{0} files have been written so far".format(i))
	os.system('./generate_gaussian_samples.py 10000 0.4 0.1 0.4 0.1 {0}'.format(i)) 


print("{0} files have been generated".format(number_of_files))

