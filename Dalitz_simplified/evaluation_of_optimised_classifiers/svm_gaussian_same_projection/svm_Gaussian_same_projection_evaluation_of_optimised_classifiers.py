import numpy as np
import math
import sys 
sys.path.insert(0,'../..')
import os
import classifier_eval_simplified
from sklearn import tree
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC


for dim in range(9,11):
	comp_file_list=[]
    
	####################################################################
	# Gaussian samples operation
	####################################################################

	for i in range(100):
		comp_file_list.append((os.environ['MLToolsDir']+"/Dalitz/gaussian_samples/gaussian_same_projection_on_each_axis/gauss_data/gaussian_same_projection_on_each_axis_redefined_{1}D_1000_0.6_0.2_0.1_{0}.txt".format(i,dim),os.environ['MLToolsDir']+"/Dalitz/gaussian_samples/gaussian_same_projection_on_each_axis/gauss_data/gaussian_same_projection_on_each_axis_redefined_{1}D_1000_0.6_0.2_0.1_1{0}.txt".format(str(i).zfill(2),dim)))

        #clf = tree.DecisionTreeClassifier('gini','best',37, 89, 1, 0.0, None)
        #clf = AdaBoostClassifier(base_estimator=tree.DecisionTreeClassifier(max_depth=2), learning_rate=0.01,n_estimators=983)
        clf = SVC(C=496.6,gamma=0.00767,probability=True, cache_size=7000)
        args=[str(dim)+ "Dgaussian_same_projection_redefined__0_1__0_1_noCPV_svm","particle","antiparticle",100,comp_file_list,1,clf,np.logspace(-2, 10, 13),np.logspace(-9, 3, 13)]
        #For nn:
        #args=[str(dim)+"Dgauss_nn","particle","antiparticle",100,comp_file_list,1,clf,np.logspace(-2, 10, 13),np.logspace(-9, 3, 13),200,6]

        ####################################################################


	classifier_eval_simplified.classifier_eval(0,0,args)


