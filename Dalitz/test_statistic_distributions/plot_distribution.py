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



distrib_file_0 = "test_statistics.2Dgauss_1000_mean_0_5_mean_0_48_sklearn_svm"
distrib_file_1 = ""

distrib_list_0 = np.loadtxt(distrib_file_0,dtype='d')
no_0 = distrib_list_0.shape[0]

U_0    = distrib_list_0[:,0]
T_0    = distrib_list_0[:,1]
D_0    = distrib_list_0[:,2]
p_Ks_0 = distrib_list_0[:,3]

print(no_0)
print(U_0)

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
	#ax1_pred_0.show()

histo_plot(U_0,10,"U value","Frequency","U distribution","U")


