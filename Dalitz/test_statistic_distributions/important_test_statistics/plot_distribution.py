#!/usr/bin/env python

#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
#
# Name:     plot_distribution.py  
#
# Author:   Constantin Weisser (weisser@mit.edu)
#
# Purpose:  If given a file with column vectors like this CvM U   CvM T   KS D    KS p
#	    then this file makes distributions of the U, T, D and p values 
#
#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#

import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np
import os

def histo_plot(U_0,abins,axlabel,aylabel,atitle,aname):
        bins_probability=np.histogram(U_0,bins=abins)[1]

        # Making a histogram of the probability predictions of the algorithm. 
        fig_pred_0= plt.figure()
        ax1_pred_0= fig_pred_0.add_subplot(1, 1, 1)
        n0, bins0, patches0 = ax1_pred_0.hist(U_0, bins=bins_probability, facecolor='red', alpha=0.5)
        ax1_pred_0.set_xlabel(axlabel)
        ax1_pred_0.set_ylabel(aylabel)
        ax1_pred_0.set_title(atitle)
        fig_pred_0.savefig(aname)
        #fig_pred_0.show()
	plt.close(fig_pred_0)	

def perm_test(T_comp,T_actual):
	#sort so largest T value comes last
	n=len(T_comp)
	#print("unsorted T_comp")
	#print(T_comp)
	T_comp=np.sort(T_comp)
	print("sorted T_comp")
	print(T_comp)
	print("T_actual")
	print(T_actual)

	return np.divide((n-1-np.searchsorted(T_comp,T_actual)),n)
		



#Plotting p values directly
print("Plotting p values directly")
distrib_file= []
for file in os.listdir("."):
        if(file.startswith("test_statistics") and not file.endswith(".png")):
                distrib_file.append(file)

# no cpv
#distrib_file= ["test_statistics.dalitz_seed_000_0_seed_100_0_sklearn_nn_rectifier10_softmax_lr0.001","test_statistics.dalitz_seed_000_0_seed_100_0_sklearn_ada_1000estimators","test_statistics.dalitz_seed_000_0_seed_100_0_sklearn_dt","test_statistics.dalitz_seed_000_0_seed_100_0_sklearn_svm","test_statistics.dalitz_seed_000_0_seed_100_0_keras_dense_activation_2_hidden"]

# cpv
#distrib_file = ["test_statistics.dalitz_seed_000_0_seed_200_1_keras_dense_activation_2_hidden","test_statistics.dalitz_seed_000_0_seed_200_1_sklearn_ada_1000estimators","test_statistics.dalitz_seed_000_0_seed_200_1_sklearn_dt","test_statistics.dalitz_seed_000_0_seed_200_1_sklearn_svm","test_statistics.dalitz_seed_000_0_seed_200_1_sklearn_nn_rectifier10_softmax_lr0.001" ]


no_files=len(distrib_file)

distrib_list = []
no = []
U = []
T = []
D = []
p_Ks = []

for i in range(no_files):
	distrib_list.append(np.loadtxt(distrib_file[i],dtype='d'))
	print(distrib_file[i])
	assert distrib_list[i].shape[0]==100
	no.append(distrib_list[i].shape[0])
	U.append(distrib_list[i][:,0])
	T.append(distrib_list[i][:,1])
	D.append(distrib_list[i][:,2])
	p_Ks.append(distrib_list[i][:,3])
	histo_plot(p_Ks[i],50,"p value","Frequency","p value distribution",distrib_file[i]+"_p_value.png")

#print(no_0)
#print(U)


#Perform permutation test
print("Performing permutation test")
distrib_file_permtest= []
for file in os.listdir("."):
        if(file.startswith("test_statistics_dalitz_seed_000_0_seed_100_0_") and not file.endswith(".png")):
                distrib_file_permtest.append(file)

distrib_list_permtest_comp = []
distrib_list_permtest_actual = []

for i in range(no_files):
        distrib_list_permtest_comp.append(np.loadtxt(distrib_file_permtest[i],dtype='d'))
	file_actual = str(np.core.defchararray.replace(distrib_file_permtest[i],"seed_100_0_","seed_200_1_"))
	print(file_actual)
	distrib_list_permtest_actual.append(np.loadtxt(file_actual,dtype='d'))
        assert distrib_list_permtest_comp[i].shape[0]==100
	assert distrib_list_permtest_actual[i].shape[0]==100
	T_comp=distrib_list_permtest_comp[i][:,1]
	T_actual=distrib_list_permtest_actual[i][:,1]
	#print(T_comp)
	#print(T_actual)	
	bins_probability=np.histogram(np.hstack((T_comp,T_actual)), bins=200)[1]

	fig_pred_comp= plt.figure()
	ax1_pred_comp= fig_pred_comp.add_subplot(1, 1, 1)
	n0, bins0, patches0 = ax1_pred_comp.hist(T_comp, bins=bins_probability, facecolor='red', alpha=0.5, label="no CPV")
	n1, bins1, patches1 = ax1_pred_comp.hist(T_actual, bins=bins_probability, facecolor='blue', alpha=0.5, label="CPV")
	#plt.axis([0.46, 0.53,0,600])
	ax1_pred_comp.legend(loc='upper left')
	ax1_pred_comp.set_xlabel('T value from CvM test')
	ax1_pred_comp.set_ylabel('Frequency')
	ax1_pred_comp.set_title(distrib_file_permtest[i]+'_T_values')
	fig_pred_comp.savefig(distrib_file_permtest[i]+'_T_values.pdf', format='pdf')

	print(perm_test(-T_comp,-T_actual))
        
	#histo_plot(p_Ks[i],50,"p value","Frequency","p value distribution",distrib_file_permtest[i]+"_p_value.png")






