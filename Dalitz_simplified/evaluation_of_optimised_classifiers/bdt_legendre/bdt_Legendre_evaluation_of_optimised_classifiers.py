import numpy as np
import math
import sys 
sys.path.insert(0,'../..')
import os
import classifier_eval_simplified
from sklearn import tree
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC


for dim in range(1,5):
	comp_file_list=[]
	contrib_string0=""
	contrib_string1=""
	contrib_string2=""
	contrib_string3=""    

	####################################################################
	# Legendre samples operation
	####################################################################


	for counter in range(dim):
		contrib_string0+= str(int((0+counter)%4))+"_0__"
		contrib_string1+= str(int((1+counter)%4))+"_0__"
		contrib_string2+= str(int((2+counter)%4))+"_0__"
		contrib_string3+= str(int((3+counter)%4))+"_0__"

	for i in range(1):

		comp_file_list.append((os.environ['MLToolsDir']+"/Dalitz/gaussian_samples/legendre/legendre_data/data_legendre_contrib0__1_0__"+contrib_string0+"contrib1__0_5__"+contrib_string1+"contrib2__2_0__"+contrib_string2+"contrib3__0_7__"+contrib_string3+"sample_{0}.txt".format(i),os.environ['MLToolsDir']+"/Dalitz/gaussian_samples/legendre/legendre_data/data_legendre_contrib0__1_0__"+contrib_string0+"contrib1__0_0__"+contrib_string1+"contrib2__2_0__"+contrib_string2+"contrib3__0_7__"+contrib_string3+"sample_{0}.txt".format(i)))

		#comp_file_list.append((os.environ['MLToolsDir']+"/Dalitz/gaussian_samples/higher_dimensional_gauss/gauss_data/data_high" +str(dim)+"Dgauss_10000_0.5_0.1_0.0_{0}.txt".format(i),os.environ['MLToolsDir']+"/Dalitz/gaussian_samples/higher_dimensional_gauss/gauss_data/data_high"+str(dim)+"Dgauss_10000_0.5_0.1_0.01_{0}.txt".format(i))) 

        #clf = tree.DecisionTreeClassifier('gini','best',37, 89, 1, 0.0, None)
        clf = AdaBoostClassifier(base_estimator=tree.DecisionTreeClassifier(max_depth=2), learning_rate=0.01,n_estimators=983)
        #clf = SVC(C=params['aC'],gamma=params['agamma'],probability=True, cache_size=7000)
        args=[str(dim)+ "Dlegendre4contrib_bdt","particle","antiparticle",100,comp_file_list,1,clf,np.logspace(-2, 10, 13),np.logspace(-9, 3, 13),0]
        #For nn:
        #args=[str(dim)+"Dgauss_nn","particle","antiparticle",100,comp_file_list,1,clf,np.logspace(-2, 10, 13),np.logspace(-9, 3, 13),params['dimof_middle'],params['n_hidden_layers']]

        ####################################################################


	classifier_eval_simplified.classifier_eval(0,0,args)


