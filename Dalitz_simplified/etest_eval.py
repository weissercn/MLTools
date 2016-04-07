


"""
This script can be used to get the p value for the Miranda method (=chi squared). It takes input files with column vectors corresponding to 
features and lables. 
"""

print(__doc__)
import os
from scipy import stats
import numpy as np
import matplotlib.pyplot as plt 
import numpy.matlib
from matplotlib.colors import Normalize

from sklearn.preprocessing import StandardScaler


##############################################################################
# Setting parameters
#

name="dalitz"
sample1_name="particle"
sample2_name="antiparticle"

C_mode=0

sigma = 0.1

comp_file_list=[]

for i in range(1,100):
	comp_file_list.append((os.environ['MLToolsDir']+"/Dalitz/dpmodel/data/data.{0}.0.txt".format(i), os.environ['MLToolsDir']+"/Dalitz/dpmodel/data/data.2{0}.1.txt".format(str(i).zfill(2))))
print(comp_file_list)
score_list=[]


##############################################################################

def weighting_function(dx):
	return np.exp(-np.square(dx)/(2*np.square(sigma)))

def distance_squared(features_0,features_1,i,j):
	dx2=0
	assert features_0.shape[1]==features_1.shape[1]
	for d in range(features_0.shape[1]):
		dx2 += np.square(features_1[j,d]-features_0[i,d])
	dx=np.sqrt(dx2)
	return weighting_function(dx)

##############################################################################
for comp_file_0,comp_file_1 in comp_file_list:

        print("Operating of files :"+comp_file_0+"   "+comp_file_1)

        #extracts data from the files
        features_0=np.loadtxt(comp_file_0,dtype='d')
        features_1=np.loadtxt(comp_file_1,dtype='d')

	#determine how many data points are in each sample
	no_0=features_0.shape[0]
	no_1=features_1.shape[0]
	
	if C_mode==0:
		T_1st_contrib=0
		print("Calculating the first contribution to T")
		for i in range(no_0):
			if(i%100==0): print(i)
			for j in range(i+1,no_0):
				T_1st_contrib += distance_squared(features_0,features_0,i,j)
		T_1st_contrib = T_1st_contrib/(no_0*(no_0-1))

		T_2nd_contrib=0
		print("Calculating the second contribution to T")
		for i in range(no_1):
			if(i%100==0): print(i)
			for j in range(i+1,no_1):
				T_2nd_contrib += distance_squared(features_1,features_1,i,j)
		T_2nd_contrib = T_2nd_contrib/(no_1*(no_1-1))


		T_3rd_contrib=0
		print("Calculating the third contribution to T")
		no_2=no_0+no_1

		for i in range(no_0):
			if(i%100==0): print(i)
			for j in range(no_1):
				T_3rd_contrib += distance_squared(features_0,features_1,i,j)
		T_3rd_contrib = T_3rd_contrib/(no_2*(no_2-1))

		T = T_1st_contrib + T_2nd_contrib +  T_3rd_contrib
	else:
		print("I need to implement the loop in C, so it runs faster")

	with open(os.path.expandvars("$MLToolsDir")+"/Dalitz/test_statistic_distributions/test_statistics_"+name+"_"+sample1_name+"_"+sample2_name+"_energy_test_"+str(sigma), "a") as test_statistics_file:
		test_statistics_file.write("{0} \t{1} \t{2} \t{3} \t{4} \n".format(0,0,0,0,T))

