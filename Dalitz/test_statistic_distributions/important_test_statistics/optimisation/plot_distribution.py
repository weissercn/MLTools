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
from __future__ import division
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np
import os

def plot_pvalue(p_Ks,param_to_be_optimised,axlabel,aylabel,atitle,aname):

	# Making a histogram of the probability predictions of the algorithm. 
        fig_pred_0= plt.figure()
        ax1_pred_0= fig_pred_0.add_subplot(1, 1, 1)
        plt.plot(param_to_be_optimised,p_Ks) 
        ax1_pred_0.set_xlabel(axlabel)
        ax1_pred_0.set_ylabel(aylabel)
        ax1_pred_0.set_title(atitle)
        fig_pred_0.savefig(aname+".png")

        plt.close(fig_pred_0)  





#Plotting p values directly
print("Plotting p values directly")
#specifying which files to operate on
distrib_name= ["test_statistics_dalitz_max_depth_optimisation_seed_000_0_seed_200_1_sklearn_dt","test_statistics_dalitz_min_sample_split_optimisation_seed_000_0_seed_200_1_sklearn_dt","test_statistics_dalitz_no_estimators_optimisation_seed_000_0_seed_200_1_sklearn_ada","test_statistics_dalitz_learning_rate_optimisation_seed_000_0_seed_200_1_sklearn_ada","test_statistics_dalitz_min_samples_split_optimisation_seed_000_0_seed_200_1_sklearn_ada","test_statistics_dalitz_no_estimators_small_optimisation_seed_000_0_seed_200_1_sklearn_ada","test_statistics_dalitz_no_estimators_smaller_optimisation_seed_000_0_seed_200_1_sklearn_ada","test_statistics_dalitz_min_samples_split_optimisation_10_estimators_seed_000_0_seed_200_1_sklearn_ada_big_scale","test_statistics_dalitz_min_samples_split_optimisation_10_estimators_seed_000_0_seed_200_1_sklearn_ada","test_statistics_dalitz_max_depth_optimisation_10_estimators_seed_000_0_seed_200_1_sklearn_ada","test_statistics_dalitz_C_optimisation_seed_000_0_seed_200_1_sklearn_svm","test_statistics_dalitz_coef0_optimisation_seed_000_0_seed_200_1_sklearn_svm"]
optimisation_param = [range(1,100),range(2,100),range(100,2001,100),[x / 10 for x in range(1,11)],range(2,100,2),range(1,301,10),range(1,21),range(2,501,10),range(20,60),range(20,60),[x/10 for x in range(1,21)], [x / 100 for x in range(0,21,2)]]
optimisation_param_name = ["max_depth","min_samples_split","no_estimators","learning_rate","min_samples_split","no_estimators","no_estimators","min_samples_split","min_samples_split","max_depth","C","coef0"]


#for file in os.listdir("."):
        #if("optimisation" in file and not "png" in file):
                #distrib_name.append(file)

#distrib_name= ["test_statistics_gaussian_samples_mean_0_5_mean_0_48_miranda_10_10"]

# no cpv
#distrib_name= ["test_statistics.dalitz_seed_000_0_seed_100_0_sklearn_nn_rectifier10_softmax_lr0.001","test_statistics.dalitz_seed_000_0_seed_100_0_sklearn_ada_1000estimators","test_statistics.dalitz_seed_000_0_seed_100_0_sklearn_dt","test_statistics.dalitz_seed_000_0_seed_100_0_sklearn_svm","test_statistics.dalitz_seed_000_0_seed_100_0_keras_dense_activation_2_hidden"]

# cpv
#distrib_name = ["test_statistics.dalitz_seed_000_0_seed_200_1_keras_dense_activation_2_hidden","test_statistics.dalitz_seed_000_0_seed_200_1_sklearn_ada_1000estimators","test_statistics.dalitz_seed_000_0_seed_200_1_sklearn_dt","test_statistics.dalitz_seed_000_0_seed_200_1_sklearn_svm","test_statistics.dalitz_seed_000_0_seed_200_1_sklearn_nn_rectifier10_softmax_lr0.001" ]


no_files=len(distrib_name)

distrib_files = []
no = []
U = []
T = []
D = []
p_Ks = []


for i in range(no_files):
	print(distrib_name[i])
	distrib_files.append(np.loadtxt(distrib_name[i],dtype='d'))
	no.append(distrib_files[i].shape[0])
	U.append(distrib_files[i][:,0])
	T.append(distrib_files[i][:,1])
	D.append(distrib_files[i][:,2])
	p_Ks.append(distrib_files[i][:,3])

	plot_pvalue(p_Ks[i],optimisation_param[i],optimisation_param_name[i],"p value from KS","Optimisation of "+optimisation_param_name[i],distrib_name[i])








