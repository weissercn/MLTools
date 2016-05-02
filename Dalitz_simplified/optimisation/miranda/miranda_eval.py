


"""
This script can be used to get the p value for the Miranda method (=chi squared). It takes input files with column vectors corresponding to 
features and lables. 
"""

print(__doc__)
import sys
sys.path.insert(0,'../..')
import os
from scipy import stats
import p_value_scoring_object
import numpy as np
import matplotlib.pyplot as plt 
import numpy.matlib
from matplotlib.colors import Normalize

from sklearn.preprocessing import StandardScaler

##############################################################################
# Setting parameters
#

name="dalitz_optimisation_miranda"
sample1_name="particle"
sample2_name="antiparticle"

shuffling_seed = 100 

single_no_bins_list=[2,3,4,5,6,7,8,9,10]

comp_file_list=[]

#comp_file_list=[(os.environ['MLToolsDir']+"/Dalitz/dpmodel/data/data_optimisation.0.0.txt",os.environ['MLToolsDir']+"/Dalitz/dpmodel/data/data_optimisation.200.1.txt")]
comp_file_list=[(os.environ['MLToolsDir']+"/Dalitz/gaussian_samples/higher_dimensional_gauss/gauss_data/data_high10Dgauss_optimisation_10000_0.5_0.1_0.0_1.txt",os.environ['MLToolsDir']+"/Dalitz/gaussian_samples/higher_dimensional_gauss/gauss_data/data_high10Dgauss_optimisation_10000_0.5_0.1_0.01_1.txt")]    

print(comp_file_list)
score_list=[]



##############################################################################
comp_file_0,comp_file_1 = comp_file_list[0]

for single_no_bins in single_no_bins_list:

	name = name + "_" + str(single_no_bins)
	print("single_no_bins : ",single_no_bins)
        print("Operating of files :"+comp_file_0+"   "+comp_file_1)

        #extracts data from the files
        features_0=np.loadtxt(comp_file_0,dtype='d')
        features_1=np.loadtxt(comp_file_1,dtype='d')

        #determine how many data points are in each sample
        no_0=features_0.shape[0]
        no_1=features_1.shape[0]

        #Give all samples in file 0 the label 0 and in file 1 the feature 1
        label_0=np.zeros((no_0,1))
        label_1=np.ones((no_1,1))

        #Create an array containing samples and features.
        data_0=np.c_[features_0,label_0]
        data_1=np.c_[features_1,label_1]

        data=np.r_[data_0,data_1]

	no_dim = data.shape[1]-1
	no_bins = [single_no_bins]*no_dim       

	np.random.shuffle(data)

	labels=data[:,-1]

	X_values= data[:,:-1]
	X_max   = np.amax(data,axis=0)[:-1] 
	X_min   = np.amin(data,axis=0)[:-1]
	X_width = (np.divide(np.subtract(X_max,X_min),no_bins))
	#print(X_width)


	setup_command_0 = "bins_sample0=np.zeros(("
	setup_command_1 = "bins_sample1=np.zeros(("
	for dim in range(no_dim):
		setup_command_0 += str(int(no_bins[dim]))+","
		setup_command_1 += str(int(no_bins[dim]))+","
	setup_command_0=setup_command_0[:-1]+"))"
	setup_command_1=setup_command_1[:-1]+"))"
	exec setup_command_0
	exec setup_command_1

	for i in range(no_0+no_1):
		#bin position
		#x_bin=int(np.floor((Xx_values[i]-Xx_min)/Xx_width))
		#y_bin=int(np.floor((Xy_values[i]-Xy_min)/Xy_width))

		pos_bins=np.floor(np.divide(np.subtract(X_values[i,:],X_min[:]),X_width[:]))
		#print(pos_bins)

		#eliminate boundary effects
		for dim in range(no_dim):
			if(pos_bins[dim]==no_bins[dim]):
				pos_bins[dim] -=1

		#if(pos_bins[0]==no_bins[0]):
			#pos_bins[0] -=1


		bin_command_0 = "bins_sample0["
		bin_command_1 = "bins_sample1["
		for dim in range(no_dim):
			bin_command_0 += str(int(pos_bins[dim]))+","
			bin_command_1 += str(int(pos_bins[dim]))+","
		bin_command_0=bin_command_0[:-1]+"]"
		bin_command_1=bin_command_1[:-1]+"]"

		#print("labels[i]: {0}".format(str(int(labels[i]))))
		#print(bin_command_0)
		if(labels[i]==0):
			#print(bin_command_0)
			#bins_sample0[y_bin,x_bin] +=1
			exec bin_command_0 + "+=1"
			#eval(bin_command_0)
			#print("labels[i]: {0}".format(str(int(labels[i]))))


		else:
			#bins_sample1[y_bin,x_bin] +=1
			exec bin_command_1 + "+=1"
			#print("labels[i]: {0}".format(str(int(labels[i]))))
	if __debug__:
		print(bins_sample0)
		print(bins_sample0[1,1])
		print(np.sum(bins_sample0))

		print(bins_sample1)
		print(bins_sample1[1,1])
		print(np.sum(bins_sample1))
	#element wise subtraction and division
	Scp2 =  np.divide(np.square(np.subtract(bins_sample1,bins_sample0)),np.add(bins_sample1,bins_sample0))
	if __debug__:
		print(Scp2)

	#nansum ignores all the contributions that are Not A Number (NAN)
	Chi2 = np.nansum(Scp2)
	if __debug__:
		print("Chi2")
		print(Chi2)
	dof=no_bins[0]
	for dim in range(1,no_dim):
		dof *= no_bins[1]
	dof-=1

	print(bins_sample0)
	print(bins_sample1)
	print("Chi2/dof : {0}".format(str(Chi2/dof)))

	pvalue= 1 - stats.chi2.cdf(Chi2,dof)

	print("pvalue : {0}".format(str(pvalue)))
	score_list.append(pvalue)

import csv
with open("miranda_10DGauss_optimisation_values", "wb") as test_statistics_file:
	test_statistics_file.write("nbins \t pvalue \n")
	writer = csv.writer(test_statistics_file, delimiter='\t', lineterminator='\n')
	writer.writerows(zip(single_no_bins_list,score_list))

fig= plt.figure()
ax1= fig.add_subplot(1, 1, 1)
ax1.plot(single_no_bins_list,score_list,'o')
print("single_no_bins_list[0]-0.1",single_no_bins_list[0]-0.1)
print("single_no_bins_list[-1]+0.1",single_no_bins_list[-1]+0.1)
plt.xlim([single_no_bins_list[0]-0.1,single_no_bins_list[-1]+0.1])
plt.ylim(-0.1,1.1)
ax1.set_xlabel("number of bins per axis")
ax1.set_ylabel("pvalue")
ax1.set_title("Miranda optimisation 10DGauss")
fig.savefig("miranda_10DGauss_optimisation_plot.png")





