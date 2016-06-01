import numpy as np
import math
import sys 
sys.path.insert(0,'../..')
import os
import classifier_eval_simplified
from sklearn import tree
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC


for dim in range(4,11):
	comp_file_list=[]
    
	####################################################################
	# Gaussian samples operation
	####################################################################

	for i in range(100):
		comp_file_list.append((os.environ['MLToolsDir']+"/Dalitz/gaussian_samples/legendre/legendre_data/data_sin1diff_5_and_5_periods{1}D_sample_{0}.txt".format(i,dim),os.environ['MLToolsDir']+"/Dalitz/gaussian_samples/legendre/legendre_data/data_sin1diff_5_and_5_periods{1}D_sample_1{0}.txt".format(str(i).zfill(2),dim)))

        #clf = tree.DecisionTreeClassifier('gini','best',37, 89, 1, 0.0, None)
        clf = AdaBoostClassifier(base_estimator=tree.DecisionTreeClassifier(max_depth=2), learning_rate=0.01,n_estimators=983)
        #clf = SVC(C=params['aC'],gamma=params['agamma'],probability=True, cache_size=7000)
        args=[str(dim)+ "Dsin1diff_5_and_5_noCPV_bdt","particle","antiparticle",100,comp_file_list,1,clf,np.logspace(-2, 10, 13),np.logspace(-9, 3, 13),0]
        #For nn:
        #args=[str(dim)+"Dgauss_nn","particle","antiparticle",100,comp_file_list,1,clf,np.logspace(-2, 10, 13),np.logspace(-9, 3, 13),params['dimof_middle'],params['n_hidden_layers']]

        ####################################################################


	classifier_eval_simplified.classifier_eval(0,0,args)


