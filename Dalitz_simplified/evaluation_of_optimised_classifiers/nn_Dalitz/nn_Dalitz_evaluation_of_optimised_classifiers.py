import numpy as np
import math
import sys 
sys.path.insert(0,'../..')
import os
import classifier_eval_simplified
from sklearn import tree
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC

comp_file_list=[]

    
####################################################################
# Dalitz operaton
####################################################################

for i in range(100):
	comp_file_list.append((os.environ['MLToolsDir']+"/Dalitz/dpmodel/data/data.{0}.0.txt".format(i), os.environ['MLToolsDir']+"/Dalitz/dpmodel/data/data.2{0}.1.txt".format(str(i).zfill(2))))
    
#clf = tree.DecisionTreeClassifier('gini','best',46, 100, 1, 0.0, None)
#clf = AdaBoostClassifier(base_estimator=tree.DecisionTreeClassifier(max_depth=2), learning_rate=0.95,n_estimators=440)
#clf = SVC(C=1.0,gamma=0.0955,probability=True, cache_size=7000)
#args=["dalitz_dt","particle","antiparticle",100,comp_file_list,1,clf,np.logspace(-2, 10, 13),np.logspace(-9, 3, 13)]
#For nn:
clf="This shouldn't be used. Keras mode"
args=["dalitz_nn_4layers_300neurons_AD","particle","antiparticle",100,comp_file_list,1,clf,np.logspace(-2, 10, 13),np.logspace(-9, 3, 13),1,300,4]

classifier_eval_simplified.classifier_eval(0,1,args)


