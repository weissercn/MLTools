import numpy as np
import math
import sys 
sys.path.insert(0,'../..')
import os
import classifier_eval_simplified
from sklearn import tree
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC


for dim in range(2,11):
	comp_file_list=[]
    
	####################################################################
	# Gaussian samples operation
	####################################################################

	for i in range(1,101):
		comp_file_list.append((os.environ['MLToolsDir']+"/Dalitz/gaussian_samples/higher_dimensional_gauss/gauss_data/data_high" +str(dim)+"Dgauss_10000_0.5_0.1_0.0_{0}.txt".format(i),os.environ['MLToolsDir']+"/Dalitz/gaussian_samples/higher_dimensional_gauss/gauss_data/data_high"+str(dim)+"Dgauss_10000_0.5_0.1_0.01_{0}.txt".format(i))) 

        clf = tree.DecisionTreeClassifier('gini','best',37, 89, 1, 0.0, None)
        #clf = AdaBoostClassifier(base_estimator=tree.DecisionTreeClassifier(max_depth=2), learning_rate=0.01,n_estimators=983)
        #clf = SVC(C=params['aC'],gamma=params['agamma'],probability=True, cache_size=7000)
        args=[str(dim)+ "Dgauss_dt","particle","antiparticle",100,comp_file_list,1,clf,np.logspace(-2, 10, 13),np.logspace(-9, 3, 13)]
        #For nn:
        #args=[str(dim)+"Dgauss_nn","particle","antiparticle",100,comp_file_list,1,clf,np.logspace(-2, 10, 13),np.logspace(-9, 3, 13),params['dimof_middle'],params['n_hidden_layers']]

        ####################################################################


	classifier_eval_simplified.classifier_eval(0,0,args)


