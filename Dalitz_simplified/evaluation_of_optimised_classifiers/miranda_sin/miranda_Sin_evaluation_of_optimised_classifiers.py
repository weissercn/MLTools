import sys 
sys.path.insert(0,'../..')
import classifier_eval_simplified


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

orig_name="sin1diff_5_and_5_periods_noCPV_p_value_distribution_miranda"
sample1_name="particle"
sample2_name="antiparticle"

shuffling_seed = 100 

single_no_bins_list=[8]
dim_list = [10]

DEBUG = False


##############################################################################
for dim_data in dim_list:
	print("We are now in "+str(dim_data) + " Dimensions")
	comp_file_list=[]
	for i in range(100):
                #comp_file_list.append((os.environ['MLToolsDir']+"/Dalitz/gaussian_samples/higher_dimensional_gauss/gauss_data/data_high" +str(dim_data)+"Dgauss_10000_0.5_0.1_0.0_{0}.txt".format(i),os.environ['MLToolsDir']+"/Dalitz/gaussian_samples/higher_dimensional_gauss/gauss_data/data_high"+str(dim_data)+"Dgauss_10000_0.5_0.1_0.01_{0}.txt".format(i))) 
		#comp_file_list.append((os.environ['MLToolsDir']+"/Dalitz/gaussian_samples/legendre/legendre_data/data_legendre_contrib0__1__10__sample_{0}.txt".format(i),os.environ['MLToolsDir']+"/Dalitz/gaussian_samples/legendre/legendre_data/data_legendre_contrib0__1__9__sample_{0}.txt".format(i)))
		comp_file_list.append((os.environ['MLToolsDir']+"/Dalitz/gaussian_samples/legendre/legendre_data/data_sin1diff_5_and_5_periods{1}D_sample_{0}.txt".format(i,dim_data),os.environ['MLToolsDir']+"/Dalitz/gaussian_samples/legendre/legendre_data/data_sin1diff_5_and_5_periods{1}D_sample_1{0}.txt".format(str(i).zfill(2),dim_data)))

	print(comp_file_list)

	for single_no_bins in single_no_bins_list:
		score_list=[]
		for comp_file_0,comp_file_1 in comp_file_list:
			name = orig_name + "_" +str(dim_data) + "D_" + str(single_no_bins)
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
			if DEBUG:
				print(bins_sample0)
				print(np.sum(bins_sample0))

				print(bins_sample1)
				print(np.sum(bins_sample1))
			#element wise subtraction and division
			numerator   = np.square(np.subtract(bins_sample1,bins_sample0))
			denominator = np.add(bins_sample1,bins_sample0)
			Scp2 =  np.divide(numerator,denominator)
			if DEBUG:
				print(Scp2)

			#nansum ignores all the contributions that are Not A Number (NAN)
			Chi2 = np.nansum(Scp2)
			if DEBUG:
				print("Chi2")
				print(Chi2)
			dof=no_bins[0]
			for dim in range(1,no_dim):
				dof *= no_bins[1]
			dof-=1
		
			pvalue= 1 - stats.chi2.cdf(Chi2,dof)

			if DEBUG:
				print(bins_sample0)
				print(bins_sample1)
				print("Chi2/dof : {0}".format(str(Chi2/dof)))

				print("pvalue : {0}".format(str(pvalue)))
			score_list.append(pvalue)

		with open(name+"_bins_p_values", "wb") as test_statistics_file:
			for score in score_list:
				test_statistics_file.write(str(score)+"\n")

		classifier_eval_simplified.histo_plot_pvalue(score_list,50,"p value","Frequency","p value distribution "+ str(single_no_bins) + " bins",name+"_bins")






