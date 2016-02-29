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

# no cpv
#distrib_file= ["test_statistics.dalitz_seed_000_0_seed_100_0_sklearn_nn_rectifier10_softmax_lr0.001","test_statistics.dalitz_seed_000_0_seed_100_0_sklearn_ada_1000estimators","test_statistics.dalitz_seed_000_0_seed_100_0_sklearn_dt","test_statistics.dalitz_seed_000_0_seed_100_0_sklearn_svm","test_statistics.dalitz_seed_000_0_seed_100_0_keras_dense_activation_2_hidden"]

# cpv
distrib_file = ["test_statistics.dalitz_seed_000_0_seed_200_1_keras_dense_activation_2_hidden","test_statistics.dalitz_seed_000_0_seed_200_1_sklearn_ada_1000estimators","test_statistics.dalitz_seed_000_0_seed_200_1_sklearn_dt","test_statistics.dalitz_seed_000_0_seed_200_1_sklearn_svm","test_statistics.dalitz_seed_000_0_seed_200_1_sklearn_nn_rectifier10_softmax_lr0.001" ]


no_files=len(distrib_file)

distrib_list = []
no = []
U = []
T = []
D = []
p_Ks = []

for i in range(no_files):
	distrib_list.append(np.loadtxt(distrib_file[i],dtype='d'))
	assert distrib_list[i].shape[0]==100
	no.append(distrib_list[i].shape[0])
	U.append(distrib_list[i][:,0])
	T.append(distrib_list[i][:,1])
	D.append(distrib_list[i][:,2])
	p_Ks.append(distrib_list[i][:,3])
	histo_plot(p_Ks[i],100,"p value","Frequency","p value distribution",distrib_file[i]+"_p_value.png")

#print(no_0)
#print(U_0)

