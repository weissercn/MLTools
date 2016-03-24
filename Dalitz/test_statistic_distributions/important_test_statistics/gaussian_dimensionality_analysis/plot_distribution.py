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
from __future__ import print_function
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np
import os

def histo_plot_pvalue(U_0,abins,axlabel,aylabel,atitle,aname):
        bins_probability=np.histogram(U_0,bins=abins)[1]

        #Finding the p values corresponding to 1,2 and 3 sigma significance.
        no_one_std_dev=sum(i < (1-0.6827) for i in U_0) 
        no_two_std_dev=sum(i < (1-0.9545) for i in U_0)
        no_three_std_dev=sum(i < (1-0.9973) for i in U_0)

        print(no_one_std_dev,no_two_std_dev,no_three_std_dev)

        #plt.rc('text', usetex=True)
        textstr = '$1\sigma=%i$\n$2\sigma=%i$\n$3\sigma=%i$'%(no_one_std_dev, no_two_std_dev, no_three_std_dev)


        # Making a histogram of the probability predictions of the algorithm. 
        fig_pred_0= plt.figure()
        ax1_pred_0= fig_pred_0.add_subplot(1, 1, 1)
        n0, bins0, patches0 = ax1_pred_0.hist(U_0, bins=bins_probability, facecolor='red', alpha=0.5)
        ax1_pred_0.set_xlabel(axlabel)
        ax1_pred_0.set_ylabel(aylabel)
        ax1_pred_0.set_title(atitle)
        plt.xlim([0,1])

	# these are matplotlib.patch.Patch properties
	props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)

	# place a text box in upper left in axes coords
	ax1_pred_0.text(0.85, 0.95, textstr, transform=ax1_pred_0.transAxes, fontsize=14,
        verticalalignment='top', bbox=props)

	fig_pred_0.savefig(aname+".png")
	#fig_pred_0.show()
        plt.close(fig_pred_0)  
	return (no_one_std_dev,no_two_std_dev,no_three_std_dev)

def perm_test(T_comp,T_actual):
	#sort so largest T value comes last
	n=len(T_comp)
	#print("unsorted T_comp")
	#print(T_comp)
	T_comp=np.sort(T_comp)
	#print("sorted T_comp")
	#print(T_comp)
	#print("T_actual")
	#print(T_actual)
	return (np.divide((n-1-np.searchsorted(T_comp,T_actual)).astype(float),n))
		



#Plotting p values directly
print("Plotting p values directly")
distrib_name= []
for file in os.listdir("."):
        if(file.startswith("test_statistics") and "dropout_5_epochs" in file and not file.endswith(".png") and not file.endswith(".pdf") ):
                distrib_name.append(file)

# no cpv
#distrib_name= ["test_statistics.dalitz_seed_000_0_seed_100_0_sklearn_nn_rectifier10_softmax_lr0.001","test_statistics.dalitz_seed_000_0_seed_100_0_sklearn_ada_1000estimators","test_statistics.dalitz_seed_000_0_seed_100_0_sklearn_dt","test_statistics.dalitz_seed_000_0_seed_100_0_sklearn_svm","test_statistics.dalitz_seed_000_0_seed_100_0_keras_dense_activation_2_hidden"]

# cpv
#distrib_name = ["test_statistics.dalitz_seed_000_0_seed_200_1_keras_dense_activation_2_hidden","test_statistics.dalitz_seed_000_0_seed_200_1_sklearn_ada_1000estimators","test_statistics.dalitz_seed_000_0_seed_200_1_sklearn_dt","test_statistics.dalitz_seed_000_0_seed_200_1_sklearn_svm","test_statistics.dalitz_seed_000_0_seed_200_1_sklearn_nn_rectifier10_softmax_lr0.001" ]

print("Using the files ",distrib_name)
no_files=len(distrib_name)

distrib_files = []
no = []
U = []
T = []
D = []
p_Ks = []

f_name='gaussian_dimensionality_analysis_nn'
f = open(f_name, 'w')
print("Writing to file ",f_name)

for i in range(no_files):
	print(distrib_name[i])
	distrib_files.append(np.loadtxt(distrib_name[i],dtype='d'))
	assert distrib_files[i].shape[0]==100
	no.append(distrib_files[i].shape[0])
	U.append(distrib_files[i][:,0])
	T.append(distrib_files[i][:,1])
	D.append(distrib_files[i][:,2])
	p_Ks.append(distrib_files[i][:,3])
	no_one_std_dev,no_two_std_dev,no_three_std_dev = histo_plot_pvalue(p_Ks[i],50,"p value","Frequency","p value distribution",distrib_name[i]+"_p_value")
	f.write(str(no_one_std_dev)+"\t"+str(no_two_std_dev)+"\t"+str(no_three_std_dev)+"\n")

f.close()
#print(no_0)
#print(U)


#Perform permutation test
print("Performing permutation test")
distrib_name_permtest= []
for file in os.listdir("."):
        if(file.startswith("test_statistics_dalitz_seed_000_0_seed_100_0_") and not ("miranda" in file)  and not file.endswith(".png") and not file.endswith(".pdf")):
                distrib_name_permtest.append(file)

distrib_files_permtest_comp = []
distrib_files_permtest_actual = []

no_files_permtest=len(distrib_name_permtest)

for i in range(no_files_permtest):
        distrib_files_permtest_comp.append(np.loadtxt(distrib_name_permtest[i],dtype='d'))
	file_actual = str(np.core.defchararray.replace(distrib_name_permtest[i],"seed_100_0_","seed_200_1_"))
	print(file_actual)
	distrib_files_permtest_actual.append(np.loadtxt(file_actual,dtype='d'))
        assert distrib_files_permtest_comp[i].shape[0]==100
	assert distrib_files_permtest_actual[i].shape[0]==100
	T_comp=distrib_files_permtest_comp[i][:,1]
	T_actual=distrib_files_permtest_actual[i][:,1]
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
	ax1_pred_comp.set_title(distrib_name_permtest[i]+'_T_values')
	fig_pred_comp.savefig(distrib_name_permtest[i]+'_T_values.pdf', format='pdf')

	p_CvM=(perm_test(T_comp,T_actual))
        
	histo_plot_pvalue(p_CvM,50,"p value","Frequency","p value distribution CvM",distrib_name_permtest[i]+"_p_value_CvM")

p_mir = np.loadtxt('gaussian_dimensionality_analysis_miranda')
p_nn  = np.loadtxt('gaussian_dimensionality_analysis_nn')
print(p_mir)
print(p_mir[:,0])

#Sorting the dimensions
dimensions= np.array([10,1,2,3,4,5,6,7,8,9])
p = dimensions.argsort()
print(p)
q = p[1::2]
print(q) 
print(p_nn[q,0])

fig = plt.figure()
ax  = fig.add_subplot(1,1,1)
ax.plot(dimensions[p],p_mir[p,0],label="Miranda 1$\sigma$",color='darkblue')
ax.plot(dimensions[p],p_mir[p,1],label="Miranda 2$\sigma$",color='blue')
ax.plot(dimensions[p],p_mir[p,2],label="Miranda 3$\sigma$",color='cyan')

ax.plot(dimensions[p],p_nn[p,0],label="Neural Net 1$\sigma$",color='darkred')
ax.plot(dimensions[p],p_nn[p,1],label="Neural Net 2$\sigma$",color='red')
ax.plot(dimensions[p],p_nn[p,2],label="Neural Net 3$\sigma$",color='tomato')

ax.set_xlabel("Number of dimensions")
ax.set_ylabel("Number of samples")
ax.set_title("Dimensionality analysis")
ax.legend(loc='lower right')
fig_name="dimensionality_analysis"
fig.savefig(fig_name)
print("Saved the figure as" , fig_name+".png")


